import warnings
from typing import Union, Optional, Sequence
from weakref import WeakSet

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike

from .context import Context
from .. import safe_import
from ...simulation import PhaseToField

GL = safe_import("OpenGL.GL", "opengl")
glfw = safe_import("glfw", "opengl")
from .patch import FrameBufferPatch, Patch, VertexArray
from ...core import PhaseSLM, Actuator, Device, Detector
from ...utilities import Transform

TimeType = Union[Quantity[u.ms], int]


class SLM(Actuator, PhaseSLM):
    """
    An OpenGL object to control a spatial light modulator connected to a graphics card.

    See :numref:`section-slms` for more information.

    """

    __slots__ = [
        "_vertex_array",
        "_frame_buffer",
        "_monitor_id",
        "_position",
        "_refresh_rate",
        "_transform",
        "_shape",
        "_window",
        "_globals",
        "_frame_buffer",
        "patches",
        "primary_patch",
        "_coordinate_system",
        "_pixel_reader",
        "_phase_reader",
        "_field_reader",
        "_context",
        "_clones",
    ]

    _active_slms = WeakSet()
    """Keep track of all active SLMs. This is done for two reasons. First, to check if we are not putting two
    full-screen SLMs on the same monitor. Second, to allow sharing the OpenGL context between all SLM windows,
    so that we can use the same Patch and Texture objects on multiple SLMs simultaneously."""

    WINDOWED = 0
    patches: list[Patch]

    def __init__(
        self,
        monitor_id: int = WINDOWED,
        shape: Optional[tuple[int, int]] = None,
        pos: tuple[int, int] = (0, 0),
        refresh_rate: Optional[Quantity[u.Hz]] = None,
        latency: TimeType = 2,
        duration: TimeType = 1,
        coordinate_system: str = "short",
        transform: Optional[Transform] = None,
    ):
        """
        Constructs a new SLM window.

        Args:
            monitor_id (int): Monitor id, see :py:attr:`~monitor_id`
            shape (tuple[int,int]): Size of the window (height, width) or resolution of the fullscreen mode.
                Default value is None (recommended for full screen windows), which will use the current resolution
                of the monitor, or a standard size of 300x300 for windowed modes.
            pos (tuple[int,int]): Windowed-mode only: (y,x)-coordinate of the top-left corner of the window.
                Default value is (0,0).
            refresh_rate (Quantity[u.Hz]): Refresh rate of the SLM.
                Ignored for windowed SLMs.
                When omitted, the current refresh rate of the monitor will be used.
                Note that OpenGL does not seem to support non-integer refresh rates.
                In these cases, it is better to set the refresh rate in the OS, and not
                explicitly specify a refresh rate.
            latency (int): Time between the vertical retrace and the start of the SLM response to the new frame,
                specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                see :py:attr:`~latency`
            duration (int): Time between the start of the SLM response to the newly presented frame, and the point
                where the SLM has stabilized.
                Specified in milliseconds (u.ms) or multiples of the frame period (unitless).
                see :py:attr:`~duration`
            transform (Transform): Global transformation matrix, see :py:attr:`~transform`.
                The `transform` determines how these vertex coordinates that make up the shape of a Patch (see
                :class:`Patch`) are mapped to the SLM window.
                By default, 'short' is used (see :attr:`transform`)

        Attributes:
            patches (list[Patch]): List of patches that are drawn on the SLM.
        """

        # construct window for displaying the SLM pattern
        SLM._init_glfw()
        self._assert_window_available(monitor_id)
        self._monitor_id = monitor_id
        self._position = pos
        (default_shape, default_rate, _) = SLM._current_mode(monitor_id)
        self._shape = default_shape if shape is None else shape
        self._refresh_rate = default_rate if refresh_rate is None else refresh_rate.to_value(u.Hz)
        self._frame_buffer = None
        self._window = None
        self._globals = -1
        self.patches = []
        self._context = None
        self._create_window()  # sets self._context, self._window and self._globals and self._frame_patch
        self._coordinate_system = coordinate_system
        self.transform = Transform() if transform is None else transform
        self._vertex_array = VertexArray()
        self._clones = WeakSet()

        # Create a single patch for displaying phase.
        # this default patch is square 1.0, and can be accessed through the 'primary_phase_patch' attribute
        # In advanced scenarios, the geometry of this patch may be modified, or it may be replaced altogether.
        self.patches.append(Patch(self._context))
        self.primary_patch = self.patches[0]
        SLM._active_slms.add(self)

        if not isinstance(duration, Quantity):
            duration = duration * self.period
        if not isinstance(latency, Quantity):
            latency = latency * self.period

        super().__init__(duration=duration, latency=latency)
        self.update()

    def _assert_window_available(self, monitor_id) -> None:
        """
        Checks if the target monitor is available for displaying an SLM window.

        Raises:
            Exception: If a full screen SLM is already present on the target monitor.
        """
        if monitor_id == SLM.WINDOWED:
            if any([slm.monitor_id == 1 for slm in SLM._active_slms if slm is not self]):
                raise RuntimeError(
                    f"Cannot create an SLM window because a full-screen SLM is already active on monitor 1"
                )
        else:
            # we cannot have multiple full screen windows on the same monitor. Also, we cannot have
            # a full screen window on monitor 1 if there are already windowed SLMs.
            if any(
                [
                    slm.monitor_id == monitor_id or (monitor_id == 1 and slm.monitor_id == SLM.WINDOWED)
                    for slm in SLM._active_slms
                    if slm is not self
                ]
            ):
                raise RuntimeError(
                    f"Cannot create a full-screen SLM window on monitor {monitor_id} because a "
                    f"window is already displayed on that monitor"
                )
            if monitor_id > len(glfw.get_monitors()):
                raise IndexError(
                    f"Monitor {monitor_id} not found, only {len(glfw.get_monitors())} monitor(s) " f"are connected."
                )

    @staticmethod
    def _current_mode(monitor_id: int):
        """Returns the current video mode resolution (height, width), refresh rate, and bit-depth of the specified
        monitor.
        For monitor_id == SLM.WINDOWED (windowed mode SLM), always returns the default window size
        of (300, 300)
        """
        if monitor_id == SLM.WINDOWED:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            shape = (300, 300)
        else:
            monitor = glfw.get_monitors()[monitor_id - 1]
            mode = glfw.get_video_mode(monitor)
            shape = (mode.size[1], mode.size[0])

        return (
            shape,
            mode.refresh_rate,
            min([mode.bits.red, mode.bits.green, mode.bits.blue]),
        )

    def _on_resize(self):
        """Updates shape and refresh rate to the actual values of the window.

        Note that these values are in pixels, which may be different from the window size because the window size is
        in screen coordinates, which may not always the same as pixels (e.g. on a retina display).

        For windowed SLMs, the refresh rate property is set to the refresh rate of the primary monitor.

        If the width, height or refresh rate differ from the requested values, or if the bit depth is less than 8,
        a warning is issued.

        This function also sets the viewport to the full window size and creates a frame buffer.
        """
        (current_size, current_rate, current_bit_depth) = SLM._current_mode(self._monitor_id)
        # verify that the bit depth is at least 8 bit
        if current_bit_depth < 8:
            warnings.warn(
                f"Bit depth is less than 8 bits " f"You may not be able to use the full phase resolution of your SLM."
            )

        # verify the refresh rate is correct
        # Then update the refresh rate to the actual value
        if int(self._refresh_rate) != current_rate:
            warnings.warn(
                f"Actual refresh rate of {current_rate} Hz does not match set rate " f"of {self._refresh_rate} Hz"
            )
        self._refresh_rate = current_rate

        # create a new frame buffer
        # re-use the lookup table if possible, otherwise create a default one ranging from 0 to 2 ** bit_depth-1.
        old_lut = self._frame_buffer.lookup_table if self._frame_buffer is not None else None
        self._frame_buffer = FrameBufferPatch(self, old_lut, current_bit_depth)
        GL.glViewport(0, 0, self._shape[1], self._shape[0])
        # tell openGL to wait for the vertical retrace when swapping buffers (it appears need to do this
        # after creating the frame buffer)
        glfw.swap_interval(1)

        # update the shape property to match the actual value of the window
        (fb_width, fb_height) = glfw.get_framebuffer_size(self._window)
        fb_shape = (fb_height, fb_width)
        if self._shape != fb_shape:
            warnings.warn(f"Actual resolution {fb_shape} does not match requested resolution {self._shape}.")
            self._shape = fb_shape

    @staticmethod
    def _init_glfw():
        """Initializes the GLFW library and sets global configuration.

        Note:
            We never de-initialize the library. This should be fine because each slm window releases its resources
            when it is destroyed. If we were to de-initialize the GLFW library (using glfw.terminate()) we run into
            trouble if the user of our library also uses glfw for something else.
        """
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

    def _create_window(self):
        """Constructs a new window and associated OpenGL context. Called by SLM.__init__()"""
        # Try to re-use an already existing OpenGL context, so that we can share textures and shaders between
        # SLM objects.
        self._context = Context(self)
        other = next(iter(SLM._active_slms), None)
        shared = other._window if other is not None else None  # noqa: ok to use _window
        SLM._active_slms.add(self)

        monitor = glfw.get_monitors()[self._monitor_id - 1] if self._monitor_id != SLM.WINDOWED else None
        glfw.window_hint(glfw.REFRESH_RATE, int(self._refresh_rate))
        self._window = glfw.create_window(self._shape[1], self._shape[0], "OpenWFS SLM", monitor, shared)
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        if monitor:  # full screen mode
            glfw.set_gamma(monitor, 1.0)
        else:  # windowed mode
            glfw.set_window_pos(self._window, self._position[1], self._position[0])

        with self._context:
            self._globals = GL.glGenBuffers(1)  # create buffer for storing globals
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)  # set clear color to black
            self._on_resize()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (height × width) of the window in pixels.

        Limitations :
            - For windowed-mode SLMs, the size cannot be modified.
            - When moving the SLM to a different monitor (see `monitor_id`), the SLM is sized to match the current
                resolution on that monitor. Note that this value may differ from the value passed as input, because the
                input value is specified in screen coordinates, whereas the reported width is in pixels.
                In this case, the original value of shape will be lost.
            - The `transform` property is not updated automatically, so if the aspect ratio changes
                the transform needs to be set again.
        """
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]):
        if self.monitor_id == SLM.WINDOWED and self._shape != value:
            glfw.set_window_size(self._window, value[1], value[0])
            self._shape = value
            self._on_resize()

    @property
    def position(self) -> tuple[int, int]:
        """
        The position of the top-left corner of the SLM window as (y, x) screen coordinates.

        Note:
            This property is ignored for full-screen SLMs.
        """
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]):
        if self.monitor_id == SLM.WINDOWED and self._position != value:
            glfw.set_window_pos(self._window, value[1], value[0])
            self._position = value

    @property
    def refresh_rate(self) -> Quantity[u.Hz]:
        """
        Refresh rate of the SLM in Hz (read only).

        Note:
            The refresh rate cannot be modified after the SLM is created. When moving
            the SLM to a different monitor (see `monitor_id`), the refresh rate is changed to the current video
            mode on that monitor. Note that OpenGL specifies this value as an integer, whereas some SLMs support
            non-integer refresh rates. It is always best to not specify the refresh rate and set the video mode
            in the operating system before creating the SLM object.
        """
        return self._refresh_rate * u.Hz

    @property
    def period(self) -> Quantity[u.ms]:
        """The period of the refresh rate in milliseconds (read only)."""
        return (1000 / self._refresh_rate) * u.ms

    @property
    def monitor_id(self) -> int:
        """
        Number of the monitor (1 for primary screen, 2 for secondary screen, etc.) for the SLM.

        Each monitor can only hold a single full-screen window.
        Use monitor_id=SLM.WINDOWED to show a windowed SLM on the primary screen
        (for debugging and monitoring purposes).
        There can be multiple windowed SLMs on the
        primary screen, but there cannot also be a fullscreen SLM on the primary screen at the same time.

        `monitor_id` can be modified at run time, in which case the current SLM window is replaced by a new window
        on a different monitor. When moving the SLM to a different window, width, height and refresh_rate are set
        """
        return self._monitor_id

    @monitor_id.setter
    def monitor_id(self, value):
        if value == self._monitor_id:
            return

        self._assert_window_available(value)
        self._monitor_id = value
        (self._shape, self._refresh_rate, _) = SLM._current_mode(value)
        monitor = glfw.get_monitors()[value - 1] if value != SLM.WINDOWED else None

        # move window to new monitor
        glfw.set_window_monitor(
            self._window,
            monitor,
            self._position[1],
            self._position[0],
            self._shape[1],
            self._shape[0],
            int(self._refresh_rate),
        )
        self._on_resize()

    def __del__(self):
        """Destructor for the SLM object. This function destroys the window and releases all resources."""
        glfw.destroy_window(self._window)

    def update(self):
        """Sends the new phase pattern to be displayed on the SLM.

        This function waits for the vsync, and returns directly after it.
        Therefore, it can be used as software synchronization to the SLM.

        Note:
            This function *does not* wait for the image to appear on the SLM.
            To wait for the image stabilization explicitly, use 'wait()'.
            However, this should rarely be needed since all Detectors
            already call wait_finished before starting a measurement.

        Note:
            At the moment, :meth:`~.SLM.update` blocks until all OpenGL commands are processed,
            and a vertical retrace occurs (i.e., the hardware signals the start of a new frame).
            This behavior may change in the future and should not be relied on.
            Instead, use the automatic synchronization mechanism to synchronize detectors with
            the SLM hardware.
        """
        with self._context:
            # first draw all patches into the frame buffer
            GL.glBindFramebuffer(
                GL.GL_FRAMEBUFFER, self._frame_buffer._frame_buffer
            )  # noqa - ok to access 'friend class'
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            for patch in self.patches:
                patch._draw()  # noqa - ok to access 'friend class'

            # then draw the frame buffer to the screen
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self._frame_buffer._draw()  # noqa - ok to access 'friend class'

            glfw.poll_events()  # process window messages

            if len(self._clones) > 0:
                self._context.__exit__(None, None, None)  # release context before updating clones
                for clone in self._clones:
                    with clone.slm._context:  # noqa
                        self._frame_buffer._draw()  # noqa - ok to access 'friend class'
                        glfw.swap_buffers(clone.slm._window)  # noqa
                self._context.__enter__()  # re-enter context

            # start 'moving' phase, then display the newly rendered image
            self._start()
            glfw.swap_buffers(self._window)

            # wait for buffer swap to complete (this should be directly after a vsync, so returning from this
            # function _should_ be synced with the vsync)
            GL.glFinish()

        # call _start again to update the _end_time_ns property,
        # since some time has passed waiting for the vsync
        self._start()

    @property
    def latency(self) -> Quantity[u.ms]:
        """Latency (a.k.a. 'idle time')

        Represents the time delay between the vertical retrace
        and the start of the SLM response to then new frame.
        """
        return self._latency

    @Device.latency.setter
    def latency(self, value: Quantity[u.ms]):
        self._latency = value.to(u.ms)

    @Device.duration.setter
    def duration(self, value: Quantity[u.ms]):
        self._duration = value.to(u.ms)

    @property
    def coordinate_system(self) -> str:
        """Specifies the base coordinate system that is used to map vertex coordinates to the SLM window.

        Possible values are 'full', 'short' and 'long'.

        'full' means that the coordinate range (-1,-1) to (1,1) is mapped to the entire SLM window.
        If the window is not square, this means that the coordinates are anisotropic.

        'short' and 'long' map the coordinate range (-1,-1) to (1,1) to a square.
        'short' means that the square is scaled to fill the short side of the SLM (introducing zero-padding at the
        edges).

        'long' means that the square is scaled to fill the long side of the SLM
        (causing part of the coordinate range to be cropped because these coordinates correspond to points outside
        the SLM window).

        For a square SLM, 'full', 'short' and 'long' are all equivalent.

        In all three cases, (-1,-1) corresponds to the top-left corner of the screen, and (1,-1) to the
        bottom-left corner. This convention is consistent with that used in numpy/matplotlib

        To further modify the mapping system, use the `transform` property.
        """
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value: str):
        if value not in ["full", "short", "long"]:
            raise ValueError(f"Unsupported coordinate system {value}")
        self._coordinate_system = value
        self.transform = self._transform  # trigger update of transform matrix on gpu

    @property
    def transform(self) -> Transform:
        """Global transformation matrix

        The transform determines how the vertex coordinates that make up the shape of a Patch (see
        :class:`Patch`) are mapped to the standard coordinate system.
        In turn, the `coordinate_system` property determines how this coordinate system is mapped to the SLM window.
        By default, this value is just the identity transformation `Transform()`.
        """
        return self._transform

    @transform.setter
    def transform(self, value: Transform):
        # first compute the basic coordinate transform
        width = self._shape[1]
        height = self._shape[0]
        if not isinstance(value, Transform):
            raise ValueError("Transform must be a Transform object")
        self._transform = value
        self._pixel_reader = None
        self._phase_reader = None
        self._field_reader = None

        # update matrix stored on the gpu
        if self._coordinate_system == "full":
            transform = self._transform
        else:
            scale_width = (width > height) == (self._coordinate_system == "short")
            if scale_width:
                root_transform = Transform(np.array(((1.0, 0.0), (0.0, height / width))))
            else:
                root_transform = Transform(np.array(((width / height, 0.0), (0.0, 1.0))))
            transform = self._transform @ root_transform

        padded = transform.opencl_matrix()
        with self._context:
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._globals)
            GL.glBufferData(GL.GL_UNIFORM_BUFFER, padded.size * 4, padded, GL.GL_STATIC_DRAW)
            GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self._globals)  # connect buffer to binding point 1

    @property
    def lookup_table(self) -> Sequence[int]:
        """Lookup table that is used to map the wrapped phase range of 0-2pi to gray values

        The gray values are represented in the range from 0 to 2**bit_depth - 1). For an 8-bit video mode, this is 0-255.
        By default, a linear lookup table is set: range(2**bit_depth - 1).

        Note: lookup table need not contain 2**bit_depth elements.
        A typical scenario is to use something like `slm.lookup_table=range(142)` to map the 0-2pi range
        to only the first 142 gray values.
        """
        return self._frame_buffer.lookup_table

    @lookup_table.setter
    def lookup_table(self, value: Sequence[int]):
        self._frame_buffer.lookup_table = value[:]

    def set_phases(self, values: ArrayLike, update=True):
        self.primary_patch.set_phases(values, update)

    @property
    def pixels(self) -> Detector:
        """Returns a 'camera' to monitor the current value of the pixels displayed on  the SLM."""
        if self._pixel_reader is None:
            self._pixel_reader = FrontBufferReader(self)
        return self._pixel_reader

    @property
    def field(self) -> Detector:
        """Returns a 'camera' to monitor the current field coming from the SLM.

        Returns:
             a detector that returns `self.amplitude * exp(1.0j * self.phases.read()`
        """
        if self._field_reader is None:
            self._field_reader = PhaseToField(self.phases)
        return self._field_reader

    @property
    def phases(self) -> Detector:
        """Returns an object to monitor the phase pattern last sent to the SLM.

        This is the pattern before applying the lookup table.
        """
        if self._phase_reader is None:
            self._phase_reader = FrameBufferReader(self)
        return self._phase_reader

    def clone(
        self,
        monitor_id: int = WINDOWED,
        shape: Optional[tuple[int, int]] = None,
        pos: tuple[int, int] = (0, 0),
    ):
        """Creates a new SLM window that mirrors the content of this SLM window.

        This is useful for demonstration and debugging purposes.
        The image in the clone window is updated automatically when the SLM
        is updated.

        Args:
            monitor_id: ID of the monitor to display the window on. Defaults to WINDOWED (0) mode
            shape: shape (height, width) of the window.
            pos: position (y, x) of the window.

        Returns:
            Returns an object that should be stored in a variable to keep the SLM window open.
            When the variable is cleared or leaves the current scope, the SLM window is closed.
            See `slm_demo.py` for an example.
        """
        clone = _Clone(slm=SLM(monitor_id=monitor_id, shape=shape, pos=pos))
        self._clones.add(clone)
        return clone


class _Clone:
    slm: SLM

    def __init__(self, slm: SLM):
        self.slm = slm


class FrontBufferReader(Detector):
    def __init__(self, slm):
        self._context = Context(slm)
        super().__init__(
            data_shape=None,
            pixel_size=None,
            duration=0.0 * u.ms,
            latency=0.0 * u.ms,
            multi_threaded=False,
        )

    @property
    def data_shape(self):
        return self._context.slm.shape

    def _fetch(self, *args, **kwargs) -> np.ndarray:
        with self._context:
            GL.glReadBuffer(GL.GL_FRONT)
            shape = self.data_shape
            data = np.empty(shape, dtype="uint8")
            GL.glReadPixels(0, 0, shape[1], shape[0], GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)
            # flip data upside down, because the OpenGL convention is to have the origin at the bottom left,
            # but we want it at the top left (like in numpy)
            return data[::-1, :]


class FrameBufferReader(Detector):
    def __init__(self, slm):
        self._context = Context(slm)
        super().__init__(
            data_shape=None,
            pixel_size=None,
            duration=0.0 * u.ms,
            latency=0.0 * u.ms,
            multi_threaded=False,
        )

    @property
    def data_shape(self):
        return self._context.slm.shape

    def _fetch(self, *args, **kwargs) -> np.ndarray:
        with self._context as slm:
            return slm._frame_buffer.get_pixels()  # noqa - ok to access 'friend class'
