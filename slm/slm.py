from OpenGL.GL import *
import numpy as np
import glfw
import warnings
import astropy.units as u
from astropy.units import Quantity
from .patch import FrameBufferPatch, Patch, VertexArray
from .geometry import fill_transform
from ..feedback import Reservation


class SLM:
    """
    Object representing a phase-only spatial light modulator. This object has many fancy functions that are important
    in setting up the SLM layers and geometry before starting the experiment. However, the algorithms only access
    a subset of functions and properties (see `SLM` protocol)
    """
    _active_monitors = np.zeros(256, 'uint32')  # keeps track of which monitors are occupied already

    MONITOR_ID_WINDOWED = 0

    def __init__(self, monitor_id=MONITOR_ID_WINDOWED, width=None, height=None, left=0, top=0, refresh_rate=None,
                 transform=None, idle_time=2, settle_time=1):
        """
        Constructs a new SLM window.
        :param monitor_id:  Monitor id, see :py:attr:`~monitor_id`
        :param width: Width of the window, or horizontal resolution of the full screen mode, see :py:attr:`~width`.
                      Defaults to None (recommended for full screen windows), which will use the current resolution
                      of the monitor, or a standard width of 300 for windowed modes.
        :param height: Height of the window, or vertical resolution of the full screen mode, see :py:attr:`~height`
                      Defaults to None (recommended for full screen windows), which will use the current resolution
                      of the monitor, or a standard height of 300 for windowed modes.
        :param left: Windowed-mode only: x-coordinate of the top-left corner of the window, see :py:attr:`~left`
        :param top:  Windowed-mode only: x-coordinate of the top-left corner of the window, see :py:attr:`~top`
        :param refresh_rate: Refresh rate (in Hz) of the SLM. Ignored for windowed SLMs, see :py:attr:`~refresh_rate`
                             Defaults to None, which will use the current refresh rate of the monitor.
        :param transform: 3x3 transformation matrix to convert from vertex coordinates to window coordinates,
                          see :py:attr:`~transform`
        :param idle_time: time between the vertical retrace and the start of the SLM response to the new frame,
                          Specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                          see :py:attr:`~idle_time`
        :param settle_time: time between the start of the SLM response to the newly presented frame, and the point
                          where the SLM has stabilized.
                          Specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                          see :py:attr:`~settle_time`
        """

        # construct window for displaying the SLM pattern
        self.patches = []
        self._monitor_id = monitor_id
        self._width = width
        self._height = height
        self._left = left
        self._top = top
        self._refresh_rate = refresh_rate
        self._reservation = Reservation()
        self._window = None
        self._globals = None  # will be filled by __create_window
        self._idle_time = None  # set by idle_time setter
        self._settle_time = None  # set by settle_time setter
        self._create_window()

        self.idle_time = idle_time
        self.settle_time = settle_time
        self.transform = transform or fill_transform(self, 'short')

        # Construct the frame buffer, this is the texture where all patches draw to. After all patches
        # finish drawing, the frame buffer itself is drawn onto the screen.
        self._frame_patch = FrameBufferPatch(self)
        self._vertex_array = VertexArray()

        # Create a single patch for displaying phase.
        # this default patch is square 1.0, and can be accessed through the 'primary_phase_patch' attribute
        # In advanced scenarios, the geometry of this patch may be modified, or it may be replaced altogether.
        self.patches = [Patch(self)]  # note: this also removes the frame patch from list of patches
        self.primary_phase_patch = self.patches[0]
        self.update()

    @property
    def monitor_id(self) -> int:
        """
        Number of the monitor (1 for primary screen, 2 for secondary screen, etc.) to display a full screen SLM
        window on. Each monitor can only hold a single full-screen window. Use MONITOR_ID_WINDOWED to show a windowed
        SLM on the primary screen (for debugging and monitoring purposes). We can have multiple windowed SLMs on the
        primary screen, but there cannot also be a full screen SLM on the primary screen at the same time.

        `monitor_id` can be modified at run time, in which case the current SLM window is replaced by a new window
         on a different monitor. When moving the SLM to a different window, width, height and refresh_rate are set
         to the current mode on that window (as if None was specified), i.e. it is only possible to select a
         display mode when first creating an SLM.
        """
        return self._monitor_id

    @monitor_id.setter
    def monitor_id(self, value):
        if value == self._monitor_id:
            return
        self.activate()
        self._monitor_id = value
        self._width = None
        self._height = None
        self._refresh_rate = None
        monitor = self._set_default_video_mode()
        glfw.set_window_monitor(self._window, monitor, self._left, self._top, self._window, self._height,
                                self._refresh_rate)
        self._set_actual_video_mode()
        glfw.set_viewport(0, 0, self._width, self._height)
        self.update()

    @property
    def width(self) -> int:
        """Width of the window in pixels. The width cannot be modified after the SLM is created. When moving
        the SLM to a different monitor (see :py:attr:`~monitor_id`), the SLM is sized to match the current resolution
        on that monitor.
        Note that this value may differ from the value passed as input, because the input value is specified in
        screen coordinates, whereas the reported width is in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Width of the window in pixels. The width can not be modified after the SLM is created. When moving
        the SLM to a different monitor (see :py:attr:`~monitor_id`), the SLM is sized to match the current resolution
        on that monitor.
        Note that this value may differ from the value passed as input, because the input value is specified in
        screen coordinates, whereas the reported width is in pixels."""
        return self._height

    @property
    def left(self) -> int:
        return self._left

    @left.setter
    def left(self, value):
        if self.monitor_id == SLM.MONITOR_ID_WINDOWED and self._left != value:
            self.activate()
            glfw.set_window_pos(self._window, value, self._top)
        self._left = value

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, value):
        if self.monitor_id == SLM.MONITOR_ID_WINDOWED and self._top != value:
            self.activate()
            glfw.set_window_pos(self._window, self._left, value)
        self._top = value

    @property
    def refresh_rate(self) -> Quantity[u.Hz]:
        """Refresh rate in Hz. The refresh rate can not be modified after the SLM is created. When moving
        the SLM to a different monitor (see :py:attr:`~monitor_id`), the refresh rate is changed to the current video
        mode on that monitor.
        Note that this value is specified by OpenGL as an integer, whereas some SLMs support non-integer refresh rates.
        It is always best to not specify the refresh_rate, and set the video mode in the OS before creating the SLM
        object."""
        return self._refresh_rate

    def _set_default_video_mode(self):
        """Set default values for the width, height and refresh rate parameters if None was specified.
        For windowed SLMs, width and height default to 300, and refresh rate is ignored.
        For full screen SLMs, width, height and refresh rate default to the current video mode.
        Returns the selected monitor (full screen SLMs) or None (windowed SLMs) for convenience.
        This function also checks if the target monitor is available, and throws an error if an SLM window is already
        present on that monitor."""
        if self._monitor_id == SLM.MONITOR_ID_WINDOWED:
            if SLM._active_monitors[1] > 0:  # prevent multiple SLM windows from opening on the same monitor
                raise Exception(f"Cannot create an SLM window because a full-screen SLM is already active on monitor 1")
            monitor = None
            self._width = self._width or 300
            self._height = self._height or 300
            self._refresh_rate = glfw.DONT_CARE * u.Hz
        else:
            # we cannot have multiple full screen windows on the same monitor. Also, we cannot have
            # a full screen window on monitor 1 if there are already windowed SLMs.
            if SLM._active_monitors[self.monitor_id] > 0 or (self.monitor_id == 1 and SLM._active_monitors[0] > 0):
                raise Exception(f"Cannot create a full-screen SLM window on monitor {self.monitor_id} because a "
                                f"window is already displayed on that monitor")
            monitor = glfw.get_monitors()[self.monitor_id - 1]
            current_mode = glfw.get_video_mode(monitor)
            self._width = self._width or current_mode.size.width
            self._height = self._height or current_mode.size.height
            self._refresh_rate = self._refresh_rate or current_mode.refresh_rate * u.Hz
        return monitor

    def _set_actual_video_mode(self):
        """Updates width, height, and refresh rate to the actual values of the window. Note that these values
        are in pixels, which may be different from the window size because the window size is in screen coordinates,
        which may not always the same as pixels (e.g. on a retina display).
        If the width, height or refresh rate differ from the requested values, or if the bit depth is less than 8,
        a warning is issued."""
        (fb_width, fb_height) = glfw.get_framebuffer_size(self._window)
        if self._width != fb_width or self._height != fb_height:
            warnings.warn(f"Actual resolution {fb_width}x{fb_height} does not match requested resolution"
                          f"{self._width}x{self._height}")
            self._width = fb_width
            self._height = fb_height

        monitor = glfw.get_window_monitor(self._window)
        if monitor:
            current_mode = glfw.get_video_mode(monitor)
            if current_mode.bits.red < 8 or current_mode.bits.green < 8 or current_mode.bits.blue < 8:
                warnings.warn(
                    f"Bit depth is less than 8 bits "
                    f"(RGB = {current_mode.bits.red},{current_mode.bits.green},{current_mode.bits.blue} bits). "
                    f"You may not be able to use the full phase resolution of your SLM.")
            if current_mode.refresh_rate != int(self._refresh_rate / u.Hz):
                warnings.warn(f"Actual refresh rate of {current_mode.refresh_rate} Hz does not match set rate "
                              f"of {self.refresh_rate}")
                self._refresh_rate = current_mode.refresh_rate * u.Hz


    def _create_window(self):
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
        glfw.window_hint(glfw.RED_BITS, 8)  # require at least 8 bits per color channel (256 gray values)
        glfw.window_hint(glfw.GREEN_BITS, 8)
        glfw.window_hint(glfw.BLUE_BITS, 8)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)  # disable retina multisampling on Mac (untested)
        glfw.window_hint(glfw.SAMPLES, 0)  # disable multisampling

        # Construct the OpenGL window. This window also acts as a 'context' which holds all resources for the window
        monitor = self._set_default_video_mode()
        glfw.window_hint(glfw.REFRESH_RATE, int(self._refresh_rate / u.Hz))
        self._window = glfw.create_window(self._width, self._height, "OpenWFS SLM", monitor, None)
        self.activate()  # Before calling any OpenGL function on the window, the context must be activated.
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        glfw.swap_interval(1)  # tell opengl to wait for the vertical retrace when swapping buffers
        self._set_actual_video_mode()
        glViewport(0, 0, self._width, self._height)

        if monitor:  # full screen mode
            glfw.set_gamma(monitor, 1.0)
        else:  # windowed mode
            glfw.set_window_pos(self._window, self.left, self.top)

        # keep track of how many SLMs we have on which monitors
        SLM._active_monitors[self.monitor_id] += 1

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

    def update(self, wait_factor=1.0, wait=True):
        self.activate()

        # first draw all patches into the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self._frame_patch.frame_buffer)
        glClear(GL_COLOR_BUFFER_BIT)
        for patch in self.patches:
            patch.draw()

        # then draw the frame buffer to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._frame_patch.draw()

        glfw.poll_events()  # process window messages

        # wait until the SLM becomes available (it may be reserved because a measurement is still pending)
        # then display the newly rendered image
        self.wait()
        glfw.swap_buffers(self._window)

        # wait for buffer swap to complete (this should be directly after a vsync, so returning from this
        # function _should_ be synced with the vsync)
        glFinish()

        # before returning, also wait until the image of the slm has stabilized
        self._reservation.reserve(self.idle_time + self.settle_time * wait_factor)

        # if wait == False, return directly. The caller can do something else and call 'wait' later to wait
        # until the image on the SLM is stable
        if wait:
            self.wait()

    @property
    def idle_time(self) -> Quantity[u.ms]:
        return self._idle_time

    @idle_time.setter
    def idle_time(self, value):
        self._idle_time = value if isinstance(value, Quantity) else value / self.refresh_rate

    @property
    def settle_time(self) -> Quantity[u.ms]:
        return self._settle_time

    @settle_time.setter
    def settle_time(self, value):
        self._settle_time = value if isinstance(value, Quantity) else value / self.refresh_rate

    def wait(self):
        self._reservation.wait()

    def reserve(self, time_seconds: Quantity[u.us]):
        self._reservation.reserve(time_seconds - self.idle_time)

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
        The matrix is a 3x3 transform matrix using homogeneous coordinates. The bottom-right element must equal 1.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        # store the transform matrix in the global data
        # Note: the transformation matrix is stored in a global buffer that is used by all shaders.
        # Unfortunately, we have to do the memory layout manually, so we add so that the vectors
        # have length 4.
        value = np.array(value, dtype=np.float32, order='C')
        if value.shape != (3, 3) or value[2, 2] != 1.0:
            raise ValueError("Transform matrix should be a 3 x 3 array, and the bottom-right element should equal 1")
        self._transform = value

        # apply padding to get 3x4 matrix, because OpenGL expects matrix to be stored with a stride of 4x4 bytes
        padded = np.append(value, np.float32([[np.nan], [np.nan], [np.nan]]), 1)

        self.activate()  # activate OpenGL context of current SLM window
        glBindBuffer(GL_UNIFORM_BUFFER, self._globals)
        glBufferData(GL_UNIFORM_BUFFER, padded.size * 4, padded, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self._globals)  # connect buffer to binding point 1

    @property
    def lookup_table(self):
        """Lookup table that is used to map the wrapped phase range 0-2pi to 8-bit color output. By default,
        this is just range(256)"""
        return self._frame_patch.lookup_table

    @lookup_table.setter
    def lookup_table(self, value):
        self._frame_patch.lookup_table = value

    @property
    def phases(self):
        return self.primary_phase_patch.phases

    @phases.setter
    def phases(self, value):
        self.primary_phase_patch.phases = value


