from OpenGL.GL import *
import numpy as np
import glfw
from .patch import FrameBufferPatch


class SLM:
    _active_monitors = np.zeros(256, 'uint32')  # keeps track of which monitors are occupied already

    def __init__(self, monitor_id=0, width=-1, height=-1, left=0, top=0, refresh_rate=-1, title="SLM",
                 transform=None):

        # construct window for displaying the SLM pattern
        self.full_screen = monitor_id != 0  # a monitor id of 0 indicates windowed mode
        self.monitor_id = monitor_id
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        self.refresh_rate = refresh_rate
        self.title = title
        self.patches = []
        self._window = None  # will be set by __create_window
        self._globals = None  # will be filled by __create_window
        self._vertex_array = None  # will be set by __create_window
        self.__create_window()

        # Set the transform matrix, use a default (where a square of 'radius' 1.0 covers shortest side of SLM)
        # if no transform is specified.
        if transform is None:  # default scaling: square
            if self.width > self.height:
                transform = [[self.height / self.width, 0.0, 0.0], [0.0, 1.0, 0.0]]
            else:
                transform = [[1.0, 0.0, 0.0], [0.0, self.width / self.height, 0.0]]
        self.transform = transform

        # Construct the frame buffer, this is the texture where all patches draw to. After all patches
        # finish drawing, the frame buffer itself is drawn onto the screen.
        self.frame_patch = FrameBufferPatch(self)
        self.patches = []  # remove frame patch from list of patches
        self.update()

    def __create_window(self):
        """Constructs a new window and associated OpenGL context. Called by SLM.__init__()"""

        ###
        # initialize the GLFW library and set global configuration. Note that we never de-initialize it. This
        # should be fine because each slm window releases its resources when it is destroyed. If we were to
        # de-initialize the GLFW library (using glfw.terminate()) we run into trouble if the user of our library also
        # uses glfw for something else.
        glfw.init()
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # Required on Mac. Doesn't hurt on Windows
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # Required on Mac. Useless on Windows
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)  # request at least opengl 4.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # Keep window on top
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)  # Disable window border
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)  # Prevent window minimization during task switch
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        # Construct the OpenGL window. This window also acts as a 'context' which holds all resources for the window
        # Before calling any OpenGL function on the window, the context must be activated.
        if self.full_screen:  # full screen mode
            # we cannot have multiple full screen windows on the same monitor. Also, we cannot have
            # a full screen window on monitor 1 if there are already windowed SLMs.
            if SLM._active_monitors[self.monitor_id] > 0 or (self.monitor_id == 1 and SLM._active_monitors[0] > 0):
                raise Exception(f"Cannot create a full-screen SLM window on monitor {self.monitor_id} because a "
                                f"window is already displayed on that monitor")

            # set defaults for size and refresh rate (use current monitor setting)
            monitor = glfw.get_monitors()[self.monitor_id - 1]
            current_mode = glfw.get_video_mode(monitor)
            if self.width == -1:
                self.width = current_mode.size.width
            if self.height == -1:
                self.height = current_mode.size.height
            if self.refresh_rate == -1:
                self.refresh_rate = current_mode.refresh_rate

            # Ask glfw to create a full screen window with specified resolution and refresh rate,
            # and at least 8 bit per pixel depth.
            glfw.window_hint(glfw.RED_BITS, 8)
            glfw.window_hint(glfw.GREEN_BITS, 8)
            glfw.window_hint(glfw.BLUE_BITS, 8)
            glfw.window_hint(glfw.REFRESH_RATE, self.refresh_rate)
            glfw.set_gamma(monitor, 1.0)
            self._window = glfw.create_window(self.width, self.height, self.title, monitor, None)

            current_mode = glfw.get_video_mode(monitor)
            (fb_width, fb_height) = glfw.get_framebuffer_size(self._window)
            if current_mode.size.width != self.width or current_mode.size.height != self.height \
                    or current_mode.refresh_rate != self.refresh_rate or current_mode.bits.red != 8 \
                    or current_mode.bits.green != 8 or current_mode.bits.blue != 8 \
                    or fb_width != self.width or fb_height != self.height:
                raise Exception(f"Could not initialize {self.width}x{self.height} full screen mode with "
                                f"bit depth of 8 and refresh rate of {self.refresh_rate}. Instead, got "
                                f"{current_mode.size.width}x{current_mode.size.height} @ {current_mode.refresh_rate} "
                                f"with bit depth of {current_mode.bits.red} "
                                f"and screen buffer size {fb_width}x{fb_height}.")

        else:  # windowed mode
            if SLM._active_monitors[1] > 0:  # prevent multiple SLM windows from opening on the same monitor
                raise Exception(f"Cannot create an SLM window because a full-screen SLM is already active on monitor 1")
            if self.width == -1:  # set default size if not specified
                self.width = 300
            if self.height == -1:
                self.height = 300
            self._window = glfw.create_window(self.width, self.height, self.title, None, None)
            glfw.set_window_pos(self._window, self.left, self.top)

        # keep track of how many SLMs we have on which monitors
        SLM._active_monitors[self.monitor_id] += 1

        # we can now start using OpenGL on the window
        self.activate()
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        glfw.swap_interval(1)  # tell opengl to wait for the vertical retrace when swapping buffers
        glViewport(0, 0, self.width, self.height)

        # Inform OpenGL about the format of the vertex data we will use.
        # Each vertex contains four float32 components:
        # x, y coordinate for vertex position. Will be transformed by the transform matrix.
        # tx, ty texture coordinates, range from 0.0, 0.0 to 1.0, 1.0 to cover the full texture
        #
        # To inform OpenGL about this format, we create vertex an array object and store format properties.
        # The elements of the vertex are available to a vertex shader as vec2 position (location 0)
        # and vec2 texture_coordinates (location 1), see vertex shader in Patch for an example.
        # All this information is bound to a binding index before use by calling glBindVertexBuffer,
        # which is done when a vertex buffer is created (see Patch).
        #
        self._vertex_array = glGenVertexArrays(1)  # no need to destroy explicitly, destroyed when window is destroyed
        glBindVertexArray(self._vertex_array)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0)  # first two float32 are screen coordinates
        glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8)  # second two are texture coordinates
        glVertexAttribBinding(0, 0)  # use binding index 0 for both attributes
        glVertexAttribBinding(1, 0)  # the attribute format can now be used with glBindVertexBuffer

        # enable primitive restart, so that we can draw multiple triangle strips with a single draw call
        glEnable(GL_PRIMITIVE_RESTART)
        glPrimitiveRestartIndex(0xFFFF)  # this is the index we use to separate individual triangle strips

        # set clear color to red for debugging
        glClearColor(1.0, 0.0, 0.0, 1.0)

        # create buffer for storing globals, and update the global transform matrix
        self._globals = glGenBuffers(1)  # no need to destroy explicitly, destroyed when window is destroyed

    def __del__(self):
        if self._window is not None:
            self.activate()
            self.patches.clear()
            glfw.destroy_window(self._window)
            SLM._active_monitors[self.monitor_id] -= 1

    def activate(self):
        """Activates the OpenGL context for this slm window. All OpenGL commands now apply to this slm"""
        glfw.make_context_current(self._window)

    def update(self):
        self.activate()

        # first draw all patches into the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_patch.frame_buffer)
        glClear(GL_COLOR_BUFFER_BIT)
        for patch in self.patches:
            patch.draw()

        # then draw the frame buffer to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.frame_patch.draw()

        glfw.poll_events()  # process window messages
        glfw.swap_buffers(self._window)

        # wait for buffer swap to complete (this should be directly after a vsync, so returning from this
        # function _should_ be synced with the vsync)
        glFinish()

    @property
    def transform(self):
        """Global transformation matrix
        for an SLM in a pupil-conjugate configuration, we use the following canonical mapping:
         - x and y axes are aligned with x and y axes of the imaging system
               (note: this compensates for a rotation along the z axis of the SLM, and any transpose or sign flips)
         - x=0, y=0 corresponds to the center of the pupil
         - sqrt(x^2 + y^2) = sin(theta), with theta the angle between the optical axis and the ray in the image plane.
         - the NA of the objective is specified separately. A default (square) patch is created that spans the NA
                exactly. If not needed, this patch can be deleted.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        # store the transform matrix in the global data
        # Note: the transformation matrix is stored in a global buffer that is used by all shaders.
        # Unfortunately, we have to do the memory layout manually, so we add so that the vectors
        # have length 4.
        value = np.array(value, dtype=np.float32)
        if value.shape != (2, 3):
            raise ValueError("Transform matrix should be a 2 x 3 array")
        self._transform = value

        padded = np.append(value, np.float32([[np.nan], [np.nan]]), 1)  # apply padding

        self.activate()  # activate OpenGL context of current SLM window
        glBindBuffer(GL_UNIFORM_BUFFER, self._globals)
        glBufferData(GL_UNIFORM_BUFFER, padded.size * 4, padded, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self._globals)  # connect buffer to binding point 1

    @property
    def lookup_table(self):
        """Lookup table that is used to map the wrapped phase range 0-2pi to 8-bit color output. By default,
        this is just range(256)"""
        return self.frame_patch.lookup_table

    @lookup_table.setter
    def lookup_table(self, value):
        self.frame_patch.lookup_table = value
