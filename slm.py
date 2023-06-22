from OpenGL.GL import *
import numpy as np
import glfw


class SLM:
    slm_count = 0

    def __init__(self, monitor_id=0, width=-1, height=-1, refresh_rate=-1, title="SLM"):
        # initialize GLFW library and set global options for window creation
        if SLM.slm_count == 0:
            glfw.init()
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # Required on Mac. Doesn't hurt on Windows
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # Required on Mac. Useless on Windows
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4) # request at least opengl 4.2
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # Keep window on top
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)  # Disable window border
            glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)  # Prevent window minimization during task switch
            glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        SLM.slm_count += 1

        # construct window for displaying the SLM pattern
        if monitor_id > 0: # full screen mode: check if a mode with the specified width, height and refresh rate is available
            monitor = glfw.get_monitors()[monitor_id - 1]

            # set defaults for size and refresh rate (use current monitor setting)
            current_mode = glfw.get_video_mode(monitor)
            if width == -1:
                width = current_mode.size.width
            if height == -1:
                height = current_mode.size.height
            if refresh_rate == -1:
                refresh_rate = current_mode.refresh_rate

            # Ask glfw to create a full screen window with specified size and refresh rate, and at least 8 bit per pixel depth
            glfw.window_hint(glfw.RED_BITS, 8)
            glfw.window_hint(glfw.GREEN_BITS, 8)
            glfw.window_hint(glfw.BLUE_BITS, 8)
            glfw.window_hint(glfw.REFRESH_RATE, refresh_rate)
            glfw.set_gamma(monitor, 1.0)
            self.window = glfw.create_window(width, height, title, monitor, None)
            current_mode = glfw.get_video_mode(monitor)
            (fb_width, fb_height) = glfw.get_framebuffer_size(self.window)
            if current_mode.size.width != width or current_mode.size.height != height\
                or current_mode.refresh_rate != refresh_rate or current_mode.bits.red != 8 or current_mode.bits.green != 8\
                or current_mode.bits.blue != 8 or fb_width != width or fb_height != height:

                raise Exception(f"Could not initialize {width}x{height} fullscreen mode with "
                    f"bit depth of 8 and refresh rate of {refresh_rate}. Instead, got "
                    f"{current_mode.size.width}x{current_mode.size.height} @ {current_mode.refresh_rate} with bit depth "
                    f"of {current_mode.bits.red} and screen buffer size {fb_width}x{fb_height}.")

        else: # windowed mode
            if width == -1:
                width = 300
            if height == -1:
                height = 300
            self.window = glfw.create_window(width, height, title, None, None)

        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN) # disable cursor
        glfw.make_context_current(self.window) # activate current opengl context (we have only one)
        glfw.swap_interval(1) # tell opengl to wait for the vertical retrace when swapping buffers
        self.width = width
        self.height = height
        self.patches = []

        # enable primitive restart, so that we can draw multiple triangle strips with a single draw call
        glEnable(GL_PRIMITIVE_RESTART)
        glPrimitiveRestartIndex(0xFFFF) # this is the index we use to separate individual triangle strips

        # create vertex array object (this object links positions in a vertex buffer to input variables in the vertex shader
        glBindVertexArray(glGenVertexArrays(1))

        # we need a 'vertex attribute array' to describe format of vertex data (same for all patches).
        # Always a float holding x,y screen coordinates and tx,ty texture coordinates (16 bytes)

        # set clear color to red for debugging
        glClearColor(1.0, 0.0, 0.0, 1.0)

        # set up globals (transformation matrix). This is a buffer of data that is shared across all programs.
        # note: we have to do the memory layout manually, so we add a padding entry (99) so that the vectors have length 4
        transform = np.array([[0.5, 0.0, 0.0, 99.0], [0.0, 1.0, 0.0, 99.0]], dtype=np.float32)
        self.globals = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.globals)
        glBufferData(GL_UNIFORM_BUFFER, transform.size * 4, transform, GL_STATIC_DRAW)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

        self.update()
        # Even with GLFW_FOCUSED = false, a full screen window will steal the focus when created. This behavior is not desired, so we
        # manually return the focus once the window is createed.
        #	    active_window = GetForegroundWindow();
        #	window.create(options["Width"], options["Height"], options["RefreshRate"], "SLM", monitor); //wrap in a smart pointer so that the window will always be destroyed when this glSLM object is destroyed
        #	SetForegroundWindow(activeWindow);
        #refreshRate = monitor.getRefreshRate();
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
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.globals)  # connect buffer holding the data to binding point 1

        for patch in self.patches:
            patch.draw()

        glfw.poll_events() # todo: only call when on the main thread
        glfw.swap_buffers(self.window)
        glFinish() # wait for buffer swap to complete (this should be directly after a vsync, so returning from this function _should_ be synced with the vsync)
        pass
