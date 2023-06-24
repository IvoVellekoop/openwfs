from OpenGL.GL import *
import numpy as np
import glfw
from patch import FrameBufferPatch

glfw_active_count = 0


def _activate_glfw():
    """Initializes the GLFW library if it is not initialized yet.
    Internally keeps a counter of active users of glfw. This counter is increased on _activate_glfw and decreased
    in _deactivate_glfw. Each call to _init_glfw should be matched by a call to _deinit_glfw.
    This function also sets global options for window and opengl context creation. If GLFW is already
    initialized, this function does nothing. """
    global glfw_active_count
    if glfw_active_count == 0:
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
    glfw_active_count += 1


def _deactivate_glfw():
    """Decreases the count of active users of GLFW. De-initializes the GLFW library if there are no more active
    users. This releases all resources and closes all windows."""
    global glfw_active_count
    glfw_active_count -= 1
    if glfw_active_count == 0:
        glfw.terminate()


class SLM:
    def __init__(self, monitor_id=0, width=-1, height=-1, x=0, y=0, refresh_rate=-1, title="SLM",
                 transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        # initialize GLFW library and set global options for window creation
        _activate_glfw()

        # construct window for displaying the SLM pattern
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.monitor_id = monitor_id
        self.refresh_rate = refresh_rate
        self.title = title
        self.__create_window(monitor_id)

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
        self._vertex_array = glGenVertexArrays(1) # no need to destroy explicitly, destroyed when window is destroyed
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
        self._globals = glGenBuffers(1) # no need to destroy explicitly, destroyed when window is destroyed
        self.transform = transform
        self.patches = []
        self.frame_patch = FrameBufferPatch(self)
        self.patches = []  # remove frame patch from list of patches
        self.update()

    def __create_window(self, title):
        if self.monitor_id > 0:  # full screen mode
            monitor = glfw.get_monitors()[self.monitor_id - 1]

            # set defaults for size and refresh rate (use current monitor setting)
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
            self.window = glfw.create_window(self.width, self.height, title, monitor, None)
            current_mode = glfw.get_video_mode(monitor)
            (fb_width, fb_height) = glfw.get_framebuffer_size(self.window)
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
            if self.width == -1:
                self.width = 300
            if self.height == -1:
                self.height = 300
            self.window = glfw.create_window(self.width, self.height, self.title, None, None)
            glfw.set_window_pos(self.window, self.x, self.y)

        self.activate()
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        glfw.swap_interval(1)  # tell opengl to wait for the vertical retrace when swapping buffers
        glViewport(0, 0, self.width, self.height)
        # Even with GLFW_FOCUSED = false, a full screen window will steal the focus when created. This behavior is
        # not desired, so we manually return the focus once the window is created. active_window =
        # GetForegroundWindow(); window.create(options["Width"], options["Height"], options["RefreshRate"], "SLM",
        # monitor); //wrap in a smart pointer so that the window will always be destroyed when this glSLM object is
        # destroyed SetForegroundWindow(activeWindow); refreshRate = monitor.getRefreshRate(); These two lines are a
        # bit silly: switch to windowed mode and back to full screen mode again. But it is a workaround for a problem
        # occurring when setting the SLM on a secondary monitor in Matlab (only in Matlab and only secondary screen)
        # glfwSetWindowMonitor(window, NULL, 0, 0, width, height, refreshRate); if (monitor.isFullScreen())
        # glfwSetWindowMonitor(window, monitor, 0, 0, width, height, refreshRate);

    def __del__(self):
        if self.window is not None:
            self.activate()
            self.patches.clear()
            glfw.destroy_window(self.window)
        _deactivate_glfw()

    def activate(self):
        """Activates the OpenGL context for this slm window. All OpenGL commands now apply to this slm"""
        glfw.make_context_current(self.window)

    def update(self):
        self.activate()
        glClear(GL_COLOR_BUFFER_BIT)

        # first draw all patches into the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_patch.frame_buffer)
        for patch in self.patches:
            patch.draw()

        # then draw the frame buffer to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.frame_patch.draw()

        glfw.poll_events()  # process window messages
        glfw.swap_buffers(self.window)

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


def enumerate_monitors():
    _activate_glfw()
    monitors = glfw.get_monitors()
    for monitor in monitors:
        print(glfw.get_monitor_name(monitor))
    _deactivate_glfw()
