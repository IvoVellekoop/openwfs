import time

from OpenGL.GL import *
import numpy as np
import glfw
import warnings
import astropy.units as u
from astropy.units import Quantity
from .patch import FrameBufferPatch, Patch, VertexArray
from .geometry import fill_transform
from weakref import WeakSet
from ..core import PhaseSLM
from typing import Union


class SLM(PhaseSLM):
    """
    Object representing a phase-only spatial light modulator. This object has many fancy functions that are important
    in setting up the SLM layers and geometry before starting the experiment. However, the algorithms only access
    a subset of functions and properties (see `SLM` protocol)
    """

    _active_slms = WeakSet()
    """"Keep track of all active SLMs. This is done for two reasons. First, to check if we are not putting two
    full-screen SLMs on the same monitor. Second, to allow sharing the OpenGL context between all SLM windows,
    so that we can use the same Patch and Texture objects on multiple SLMs simultaneously."""

    WINDOWED = 0

    def __init__(self, monitor_id=WINDOWED, shape=None, pos=(0, 0), refresh_rate=glfw.DONT_CARE * u.Hz,
                 transform=None, latency=2, settle_time=1, wavelength=None, lut_generator=None):
        """
        Constructs a new SLM window.

        Args:
            monitor_id:  Monitor id, see :py:attr:`~monitor_id`
            shape(tuple[int,int]):
                Size of the window (height, width) or resolution of the fullscreen mode.
                Default valie is None (recommended for full screen windows), which will use the current resolution
                of the monitor, or a standard size of 300x300 for windowed modes.
            pos(tuple[int,int]):
                Windowed-mode only: (y,x)-coordinate of the top-left corner of the window.
                Default value (0,0).
            refresh_rate(Quantity[u.Hz]): Refresh rate of the SLM.
                Ignored for windowed SLMs.
                When omitted, the current refresh rate of the monitor will be used.
                Note that OpenGL does not seem to support non-integer refresh rates.
                In these cases, it is better to set the refresh rate in the OS, and don't
                explicitly specify a refresh rate.
            transform: 3x3 transformation matrix to convert from vertex coordinates to window coordinates,
                see :py:attr:`~transform`
            latency: time between the vertical retrace and the start of the SLM response to the new frame,
                       Specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                       see :py:attr:`~idle_time`
            settle_time: time between the start of the SLM response to the newly presented frame, and the point
                       where the SLM has stabilized.
                       Specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                       see :py:attr:`~settle_time`
            wavelength: wavelength in nanometers (u.ns). Use in combination with lut_generator
            lut_generator: function taking a wavelength in nanometers and returning a lookup table. This function
                       is called every time the wavelength of the slm object is set, so that the lookup table is
                       be adapted to the wavelength of the light source.
        """
        super().__init__()

        # construct window for displaying the SLM pattern
        self.patches = []
        self.lut_generator = lut_generator or (lambda λ: np.arange(0, 256) / 255)
        self._monitor_id = monitor_id
        self._shape = shape
        self._pos = pos
        self._refresh_rate = refresh_rate
        self._window = None
        self._globals = None  # will be filled by __create_window
        self._wavelength = wavelength
        self._create_window()
        SLM._active_slms.add(self)

        self.latency = latency
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
        primary screen, but there cannot also be a fullscreen SLM on the primary screen at the same time.

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
        self._shape = None
        self._refresh_rate = 0.0 * u.Hz
        monitor = self._set_default_video_mode()
        glfw.set_window_monitor(self._window, monitor, self._pos[1], self._pos[0], self._shape[1], self._shape[0],
                                int(self._refresh_rate / u.Hz))
        self._set_actual_video_mode()
        glViewport(0, 0, self._shape[1], self._shape[0])
        # construct new frame buffer object, re-use LUT
        lut = self.lookup_table
        self._frame_patch = FrameBufferPatch(self)
        self.lookup_table = lut
        self.update()

    @property
    def shape(self) -> tuple:
        """Shape (height x width) of the window in pixels.

        The size cannot be modified after the SLM is created. When moving
        the SLM to a different monitor (see :py:attr:`~monitor_id`), the SLM is sized to match the current resolution
        on that monitor.
        Note that this value may differ from the value passed as input, because the input value is specified in
        screen coordinates, whereas the reported width is in pixels."""
        return self._shape

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        if self.monitor_id == SLM.WINDOWED and self._pos != value:
            self.activate()
            glfw.set_window_pos(self._window, value[1], value[0])
        self._pos = value

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
        if self._monitor_id == SLM.WINDOWED:
            for slm in SLM._active_slms:
                if slm is not self and slm.monitor_id == 1:
                    raise Exception(f"Cannot create an SLM window because a full-screen SLM is already active on "
                                    f"monitor 1")
            monitor = None
            self._shape = self._shape or (300, 300)
            self._refresh_rate = glfw.DONT_CARE * u.Hz
        else:
            # we cannot have multiple full screen windows on the same monitor. Also, we cannot have
            # a full screen window on monitor 1 if there are already windowed SLMs.
            for slm in SLM._active_slms:
                if slm is not self and slm.monitor_id == self._monitor_id or \
                        (self._monitor_id == 1 and slm.monitor_id == SLM.WINDOWED):
                    raise Exception(f"Cannot create a full-screen SLM window on monitor {self.monitor_id} because a "
                                    f"window is already displayed on that monitor")
            monitor = glfw.get_monitors()[self.monitor_id - 1]
            current_mode = glfw.get_video_mode(monitor)
            self._shape = self._shape or (current_mode.size.height, current_mode.size.width)
            if self._refresh_rate.to_value(u.Hz) == glfw.DONT_CARE:
                self._refresh_rate = current_mode.refresh_rate * u.Hz
        return monitor

    def _set_actual_video_mode(self):
        """Updates width, height, and refresh rate to the actual values of the window. Note that these values
        are in pixels, which may be different from the window size because the window size is in screen coordinates,
        which may not always the same as pixels (e.g. on a retina display).
        If the width, height or refresh rate differ from the requested values, or if the bit depth is less than 8,
        a warning is issued."""
        (fb_width, fb_height) = glfw.get_framebuffer_size(self._window)
        fb_shape = (fb_height, fb_width)
        if self._shape != fb_shape:
            raise Exception("aa")
            warnings.warn(f"Actual resolution {fb_shape} does not match requested resolution {self._shape}.")
            self._shape = fb_shape

        monitor = glfw.get_window_monitor(self._window)
        if monitor:
            current_mode = glfw.get_video_mode(monitor)
            if current_mode.bits.red < 8 or current_mode.bits.green < 8 or current_mode.bits.blue < 8:
                warnings.warn(
                    f"Bit depth is less than 8 bits "
                    f"(RGB = {current_mode.bits.red},{current_mode.bits.green},{current_mode.bits.blue} bits). "
                    f"You may not be able to use the full phase resolution of your SLM.")
            if (self.refresh_rate != glfw.DONT_CARE * u.Hz and
                    current_mode.refresh_rate != int(self._refresh_rate / u.Hz)):
                warnings.warn(f"Actual refresh rate of {current_mode.refresh_rate} Hz does not match set rate "
                              f"of {self.refresh_rate}")
        else:
            current_mode = glfw.get_video_mode(glfw.get_primary_monitor())
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
        shared = next(iter(SLM._active_slms), self)._window
        self._window = glfw.create_window(self._shape[1], self._shape[0], "OpenWFS SLM", monitor, shared)
        self.activate()  # Before calling any OpenGL function on the window, the context must be activated.
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        glfw.swap_interval(1)  # tell opengl to wait for the vertical retrace when swapping buffers
        self._set_actual_video_mode()
        glViewport(0, 0, self._shape[1], self._shape[0])

        if monitor:  # full screen mode
            glfw.set_gamma(monitor, 1.0)
        else:  # windowed mode
            glfw.set_window_pos(self._window, self._pos[1], self._pos[0])

        # set clear color to black
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # create buffer for storing globals, and update the global transform matrix
        self._globals = glGenBuffers(1)  # no need to destroy explicitly, destroyed when window is destroyed

    def __del__(self):
        if self._window is not None:
            self.activate()
            self.patches.clear()
            glfw.destroy_window(self._window)

    def activate(self):
        """Activates the OpenGL context for this slm window. All OpenGL commands now apply to this slm"""
        glfw.make_context_current(self._window)

    def update(self, wait_factor=1.0):
        """Sends the new phase pattern to be displayed on the SLM.
        Args:
            wait_factor.
                Time to allow for stabilization of the image, relative to the settle_time property.
                Values higher than 1.0 allow extra stabilization time before measurements are made.

        Note:
            This function waits for the vsync, and returns directly after it.
            Therefore, it can be used as software synchronization to the SLM.
        Note:
            This function *does not* wait for the image to appear on the SLM.
            To wait for the image stabilization explicitly, use 'wait_finished'.
            However, this should rarely be needed since all Detectors
            already call wait_finished before starting a measurement.
        """
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

        # start 'moving' phase, then display the newly rendered image
        self._start()
        glfw.swap_buffers(self._window)

        # wait for buffer swap to complete (this should be directly after a vsync, so returning from this
        # function _should_ be synced with the vsync)
        glFinish()

        # update the start time, since some time has passed waiting for the vsync
        # also adjust the start time for the wait factor, so that wait_finished can still wait
        # until _start_time_ns + ._duration
        self._start_time_ns = time.time_ns() + np.rint((self._duration * (1.0 - wait_factor)).to_value(u.ns))

    @property
    def latency(self) -> Quantity[u.ms]:
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value if isinstance(value, Quantity) else value / self.refresh_rate

    @property
    def settle_time(self) -> Quantity[u.ms]:
        return self._duration

    @settle_time.setter
    def settle_time(self, value):
        self._duration = value if isinstance(value, Quantity) else value / self.refresh_rate

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

    def set_phases(self, values: Union[np.ndarray, float], update=True):
        self.primary_phase_patch.phases = values
        if update:
            self.update()

    def get_pixels(self, type='phase'):
        """Read back the pixels currently displayed on the SLM."""
        if type == 'gray_value':
            self.activate()
            glReadBuffer(GL_FRONT)
            data = np.empty(self.shape, dtype='uint8')
            glReadPixels(0, 0, self._shape[1], self._shape[0], GL_RED, GL_UNSIGNED_BYTE, data)
            return data
        if type == 'phase':
            return self._frame_patch.get_pixels()

        raise ValueError(f"Unsupported pixel type {type}")

    @property
    def wavelength(self) -> Quantity[u.nm]:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self.lookup_table = self.lut_generator(value)
