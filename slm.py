from OpenGL.GL import *
import numpy as np
import glfw
from patch import Patch


class SLM:
    slm_count = 0

    def __init__(self, monitor_id=0, width=-1, height=-1, refresh_rate=-1, title="SLM", transform=None,
                 numerical_aperture=0.8):
        # initialize GLFW library and set global options for window creation
        if SLM.slm_count == 0:
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
        SLM.slm_count += 1

        # construct window for displaying the SLM pattern
        if monitor_id > 0:  # full screen mode
            monitor = glfw.get_monitors()[monitor_id - 1]

            # set defaults for size and refresh rate (use current monitor setting)
            current_mode = glfw.get_video_mode(monitor)
            if width == -1:
                width = current_mode.size.width
            if height == -1:
                height = current_mode.size.height
            if refresh_rate == -1:
                refresh_rate = current_mode.refresh_rate

            # Ask glfw to create a full screen window with specified resolution and refresh rate,
            # and at least 8 bit per pixel depth.
            glfw.window_hint(glfw.RED_BITS, 8)
            glfw.window_hint(glfw.GREEN_BITS, 8)
            glfw.window_hint(glfw.BLUE_BITS, 8)
            glfw.window_hint(glfw.REFRESH_RATE, refresh_rate)
            glfw.set_gamma(monitor, 1.0)
            self.window = glfw.create_window(width, height, title, monitor, None)
            current_mode = glfw.get_video_mode(monitor)
            (fb_width, fb_height) = glfw.get_framebuffer_size(self.window)
            if current_mode.size.width != width or current_mode.size.height != height \
                    or current_mode.refresh_rate != refresh_rate or current_mode.bits.red != 8\
                    or current_mode.bits.green != 8 or current_mode.bits.blue != 8\
                    or fb_width != width or fb_height != height:
                raise Exception(f"Could not initialize {width}x{height} fullscreen mode with "
                                f"bit depth of 8 and refresh rate of {refresh_rate}. Instead, got "
                                f"{current_mode.size.width}x{current_mode.size.height} @ {current_mode.refresh_rate} "
                                f"with bit depth of {current_mode.bits.red} "
                                f"and screen buffer size {fb_width}x{fb_height}.")

        else:  # windowed mode
            if width == -1:
                width = 300
            if height == -1:
                height = 300
            self.window = glfw.create_window(width, height, title, None, None)

        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        glfw.make_context_current(self.window)  # activate current opengl context (we have only one)
        glfw.swap_interval(1)  # tell opengl to wait for the vertical retrace when swapping buffers
        self.width = width
        self.height = height

        # enable primitive restart, so that we can draw multiple triangle strips with a single draw call
        glEnable(GL_PRIMITIVE_RESTART)
        glPrimitiveRestartIndex(0xFFFF)  # this is the index we use to separate individual triangle strips

        # create vertex array object. This object links positions in a vertex buffer to input
        # variables in the vertex shader. We don't do anything with it, but it is needed by opengl
        self.vertex_array = glGenVertexArrays(1)
        glBindVertexArray(self.vertex_array)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0)  # first two float32 are screen coordinates
        glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8)  # second two are texture coordinates
        glVertexAttribBinding(0, 0) # use binding index 0 for both attributes
        glVertexAttribBinding(1, 0) # the attribute format can now be used with glBindVertexBuffer
        # glBindVertexArray(0)   # keep bound, we only have one

        # we need a 'vertex attribute array' to describe format of vertex data (same for all patches).
        # Always a float holding x,y screen coordinates and tx,ty texture coordinates (16 bytes)

        # set clear color to red for debugging
        glClearColor(1.0, 0.0, 0.0, 1.0)

        # set up the global transformation matrix
        # for an SLM in a pupil-conjugate configuration, we use the following canonical mapping:
        # - x and y axes are aligned with x and y axes of the imaging system
        #       (note: this compensates for a rotation along the z axis of the SLM, and any transpose or sign flips)
        # - x=0, y=0 corresponds to the center of the pupil
        # - sqrt(x^2 + y^2) = sin(theta), with theta the angle between the optical axis and the ray in the image plane.
        # - the NA of the objective is specified separately. A default (square) patch is created that spans the NA
        #        exactly. If not needed, this patch can be deleted.
        #
        # Note: the transformation matrix is stored in a global buffer that is used by all shaders.
        # Unfortunately, we have to do the memory layout manually, so we add so that the vectors
        # have length 4
        if transform is None:
            transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        transform = np.array(transform, dtype=np.float32)
        if transform.shape != (2, 3):
            raise ValueError("Transform matrix should be a 2 x 3 array")
        transform = np.append(transform, np.float32([[np.nan], [np.nan]]), 1)  # apply padding

        self.globals = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.globals)
        glBufferData(GL_UNIFORM_BUFFER, transform.size * 4, transform, GL_STATIC_DRAW)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

        # set default patch
        rectangle = [
            [[-numerical_aperture, -numerical_aperture, 0.0, 0.0], [numerical_aperture, -numerical_aperture, 1.0, 0.0]],
            [[-numerical_aperture, numerical_aperture, 0.0, 1.0], [numerical_aperture, numerical_aperture, 1.0, 1.0]]]

        self.patches = [Patch(rectangle)]

        self.update()
        # Even with GLFW_FOCUSED = false, a full screen window will steal the focus when created. This behavior is not desired, so we
        # manually return the focus once the window is createed.
        #	    active_window = GetForegroundWindow();
        #	window.create(options["Width"], options["Height"], options["RefreshRate"], "SLM", monitor); //wrap in a smart pointer so that the window will always be destroyed when this glSLM object is destroyed
        #	SetForegroundWindow(activeWindow);
        # refreshRate = monitor.getRefreshRate();
        # These two lines are a bit silly: switch to windowed mode and back to full screen mode again. But it is a workaround for a problem occuring when setting the SLM on a secondary monitor in Matlab
        # (only in Matlab and only secondary screen)
        # glfwSetWindowMonitor(window, NULL, 0, 0, width, height, refreshRate);
        # if (monitor.isFullScreen())
        #	glfwSetWindowMonitor(window, monitor, 0, 0, width, height, refreshRate);

    def __del__(self):
        if self.window is not None:
            glfw.destroy_window(self.window)

        SLM.slm_count -= 1
        if SLM.slm_count == 0:
            glfw.terminate()

    def enumerate_monitors(self):
        monitors = glfw.get_monitors()
        for monitor in monitors:
            print(glfw.get_monitor_name(monitor))

    def update(self):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1,
                         self.globals)  # connect buffer holding the globals data to binding point 1
        glBindVertexArray(self.vertex_array)

        for patch in self.patches:
            patch.draw()

        glfw.poll_events()  # todo: only call when on the main thread
        glfw.swap_buffers(self.window)
        glBindVertexArray(0)
        glFinish()  # wait for buffer swap to complete (this should be directly after a vsync, so returning from this function _should_ be synced with the vsync)
        pass
