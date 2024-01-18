from OpenGL.GL import *
import numpy as np
import glfw
import warnings
import astropy.units as u
from numpy.typing import ArrayLike
from typing import Union, Optional, List, Sequence
from astropy.units import Quantity
from .patch import FrameBufferPatch, Patch, VertexArray
from weakref import WeakSet
from ..core import PhaseSLM, Actuator
from ..utilities import Transform

TimeType = Union[Quantity[u.ms], int]


class SLM(Actuator, PhaseSLM):
    """
    An OpenGL object to control a spatial light modulator connected to a graphics card.

    The SLM can be created in windowed mode (useful for debugging), or full screen.
    It is possible to have multiple windowed SLMs on the same monitor, but only one full-screen SLM per monitor.

    An SLM holds a list of patches, which are shapes that can be drawn on the SLM.
    Each of these patches can have a different texture, which is a 2D array of values that are drawn on the SLM.
    In addition, each patch can have its own geometry, which determines how the texture is mapped onto the SLM window.
    For more information, see the documentation of the Patch class.

    If an SLM holds multiple patches, the patches are drawn in the order they are present in the `patches` list.
    If patches overlap, the pixels of the previous patch are either overwritten (when the `additive_blend` property
    of the patch is False), or added to the phase values of the previous patch (when `additive_blend` is True).

    This way, a large range of use cases is enabled, including:
    - Drawing a single square patch with a single texture (the default).
    - Mapping the phase values to a disk, with an effective resolution depending on the distance to the center of the disk (see `geometry.disk`).
    - Applying an additive patch (an offset layer) that corrects for system aberrations.
    - etc.

    The SLM object also holds a lookup table that maps the phase values to gray values.
    By default, this is a linear table that maps wrapped phase values from 0-2pi to gray values from 0-255.
    This table can be modified to correct for the non-linear response of the hardware, or to scale the range
    of gray values that is used.

    Attributes:
        patches (list[openwfs.slm.Patch]): List of patches that are drawn on the SLM.

    """
    __slots__ = ['_vertex_array', '_frame_buffer', '_monitor_id', '_pos', '_latency', '_duration', '_refresh_rate',
                 '_transform', '_shape', '_window', '_globals', '_frame_buffer', 'patches', 'primary_patch']

    _active_slms = WeakSet()
    """Keep track of all active SLMs. This is done for two reasons. First, to check if we are not putting two
    full-screen SLMs on the same monitor. Second, to allow sharing the OpenGL context between all SLM windows,
    so that we can use the same Patch and Texture objects on multiple SLMs simultaneously."""

    WINDOWED = 0

    transform: Transform
    patches: List[Patch]
    _monitor_id: int
    _pos: tuple[int, int]
    _latency: TimeType
    _duration: TimeType
    _refresh_rate: Quantity[u.Hz]
    _shape: tuple[int, int]
    _window: object
    _globals: int
    _frame_buffer: 'Optional[FrameBufferPatch]'
    _vertex_array: VertexArray

    def __init__(self, monitor_id: int = WINDOWED, shape: Optional[tuple[int, int]] = None,
                 pos: tuple[int, int] = (0, 0), refresh_rate=None,
                 latency: TimeType = 2, duration: TimeType = 1, transform: Union[Transform | str] = 'short'):
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
        """

        # construct window for displaying the SLM pattern
        SLM._init_glfw()
        self._assert_window_available(monitor_id)
        self._monitor_id = monitor_id
        self._pos = pos
        self._latency = latency
        self._duration = duration
        (default_shape, default_rate, _) = SLM._current_mode(monitor_id)
        self._shape = default_shape if shape is None else shape
        self._refresh_rate = default_rate if refresh_rate is None else refresh_rate
        self._frame_buffer = None
        self._window = None
        self._globals = -1
        self.patches = []
        self._create_window()  # sets self._window and self._globals and self._frame_patch
        self.transform = transform
        self._vertex_array = VertexArray()

        # Create a single patch for displaying phase.
        # this default patch is square 1.0, and can be accessed through the 'primary_phase_patch' attribute
        # In advanced scenarios, the geometry of this patch may be modified, or it may be replaced altogether.
        self.patches.append(Patch(self))
        self.primary_patch = self.patches[0]
        SLM._active_slms.add(self)
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
                    f"Cannot create an SLM window because a full-screen SLM is already active on monitor 1")
        else:
            # we cannot have multiple full screen windows on the same monitor. Also, we cannot have
            # a full screen window on monitor 1 if there are already windowed SLMs.
            if any([slm.monitor_id == monitor_id or
                    (monitor_id == 1 and slm.monitor_id == SLM.WINDOWED)
                    for slm in SLM._active_slms if slm is not self]):
                raise RuntimeError(f"Cannot create a full-screen SLM window on monitor {monitor_id} because a "
                                   f"window is already displayed on that monitor")
            if monitor_id > len(glfw.get_monitors()):
                raise IndexError(f"Monitor {monitor_id} not found, only {len(glfw.get_monitors())} monitor(s) "
                                 f"are connected.")

    @staticmethod
    def _current_mode(monitor_id: int):
        """Returns the current video mode resolution (height, width), refresh rate and bit depth of the specified
        monitor. For monitor_id == SLM.WINDOWED (windowed mode SLM), always returns the default window size
        of (300, 300)"""
        if monitor_id == SLM.WINDOWED:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            return ((300, 300), mode.refresh_rate * u.Hz,
                    min([mode.bits.red, mode.bits.green, mode.bits.blue]))
        else:
            monitor = glfw.get_monitors()[monitor_id - 1]
            mode = glfw.get_video_mode(monitor)
            return ((mode.size[1], mode.size[0]), mode.refresh_rate * u.Hz,
                    min([mode.bits.red, mode.bits.green, mode.bits.blue]))

    def _on_resize(self):
        """Updates shape and refresh rate to the actual values of the window.

        Note that these values are in pixels, which may be different from the window size because the window size is
        in screen coordinates, which may not always the same as pixels (e.g. on a retina display).

        For windowed SLMs, the refresh rate property is set to the refresh rate of the primary monitor.

        If the width, height or refresh rate differ from the requested values, or if the bit depth is less than 8,
        a warning is issued.

        This function also sets the viewport to the full window size and creates a frame buffer.
        """
        # create a new frame buffer, re-use the old one if one was present, otherwise use a default of range(256)
        # re-use the lookup table if possible, otherwise create a default one ranging from 0 to 255.
        old_lut = self._frame_buffer.lookup_table if self._frame_buffer is not None else range(256)
        self._frame_buffer = FrameBufferPatch(self, old_lut)
        glViewport(0, 0, self._shape[1], self._shape[0])
        # tell openGL to wait for the vertical retrace when swapping buffers (it appears need to do this
        # after creating the frame buffer)
        glfw.swap_interval(1)

        # update the shape property to match the actual value of the window
        (fb_width, fb_height) = glfw.get_framebuffer_size(self._window)
        fb_shape = (fb_height, fb_width)
        if self._shape != fb_shape:
            warnings.warn(f"Actual resolution {fb_shape} does not match requested resolution {self._shape}.")
            self._shape = fb_shape

        (current_size, current_rate, current_bit_depth) = SLM._current_mode(self._monitor_id)

        # verify bit depth is at least 8 bit
        if current_bit_depth < 8:
            warnings.warn(
                f"Bit depth is less than 8 bits "
                f"You may not be able to use the full phase resolution of your SLM.")

        self.activate()

        # verify the refresh rate is correct
        # Then update the refresh rate to the actual value
        if int(self.refresh_rate.to_value(u.Hz)) != current_rate.to_value(u.Hz):
            warnings.warn(f"Actual refresh rate of {current_rate} Hz does not match set rate "
                          f"of {self.refresh_rate}")
        self._refresh_rate = current_rate

    @staticmethod
    def _init_glfw():
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

    def _create_window(self):
        """Constructs a new window and associated OpenGL context. Called by SLM.__init__()"""
        # Try to re-use an already existing OpenGL context, so that we can share textures and shaders between
        # SLM objects.
        shared = next((slm._window for slm in SLM._active_slms), None)
        monitor = glfw.get_monitors()[self._monitor_id - 1] if self._monitor_id != SLM.WINDOWED else None

        glfw.window_hint(glfw.REFRESH_RATE, int(self._refresh_rate.to_value(u.Hz)))
        self._window = glfw.create_window(self._shape[1], self._shape[0], "OpenWFS SLM", monitor, shared)
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)  # disable cursor
        if monitor:  # full screen mode
            glfw.set_gamma(monitor, 1.0)
        else:  # windowed mode
            glfw.set_window_pos(self._window, self._pos[1], self._pos[0])

        self.activate()  # Before calling any OpenGL function on the window, the context must be activated.
        self._globals = glGenBuffers(1)  # create buffer for storing globals
        glClearColor(0.0, 0.0, 0.0, 1.0)  # set clear color to black
        self._on_resize()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the shape (height x width) of the window in pixels.

        Returns:
            Tuple[int, int]: The shape of the window in pixels.

        Note:
            For windowed-mode SLMs, the The size cannot be modified after the SLM is created. When moving
            the SLM to a different monitor (see `monitor_id`), the SLM is sized to match the current resolution
            on that monitor. Note that this value may differ from the value passed as input, because the input value
            is specified in screen coordinates, whereas the reported width is in pixels.
        """
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]):
        if self.monitor_id == SLM.WINDOWED and self._shape != value:
            # self.activate()
            glfw.set_window_size(self._window, value[1], value[0])
            self._shape = value
            self._on_resize()

    @property
    def pos(self) -> tuple[int, int]:
        """
        Returns the position of the top-left corner of the SLM window.

        Returns:
            Tuple[int, int]: The position of the top-left corner of the SLM window as (y, x) coordinates.

        Note:
            The position can be modified by assigning a new tuple of (y, x) coordinates to the `pos` property.
            When moving the SLM window to a different position, the SLM is activated and the window position is updated.
        """
        return self._pos

    @pos.setter
    def pos(self, value: tuple[int, int]):
        if self.monitor_id == SLM.WINDOWED and self._pos != value:
            self.activate()
            glfw.set_window_pos(self._window, value[1], value[0])
            self._pos = value

    @property
    def refresh_rate(self) -> Quantity[u.Hz]:
        """
        Returns the refresh rate of the SLM in Hz.

        Returns:
            Quantity[u.Hz]: The refresh rate of the SLM.

        Note:
            The refresh rate cannot be modified after the SLM is created. When moving
            the SLM to a different monitor (see `monitor_id`), the refresh rate is changed to the current video
            mode on that monitor. Note that OpenGL specifies this value as an integer, whereas some SLMs support
            non-integer refresh rates. It is always best to not specify the refresh rate and set the video mode
            in the operating system before creating the SLM object.
        """
        return self._refresh_rate

    @property
    def monitor_id(self) -> int:
        """
        Number of the monitor (1 for primary screen, 2 for secondary screen, etc.) to display a full screen SLM
        window on. Each monitor can only hold a single full-screen window. Use MONITOR_ID_WINDOWED to show a windowed
        SLM on the primary screen (for debugging and monitoring purposes). We can have multiple windowed SLMs on the
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
        glfw.set_window_monitor(self._window, monitor, self._pos[1], self._pos[0], self._shape[1], self._shape[0],
                                int(self._refresh_rate.to_value(u.Hz)))
        self._on_resize()

    def __del__(self):
        if self._window is not None:
            self.activate()
            self.patches.clear()
            glfw.destroy_window(self._window)

    def activate(self):
        """Activates the OpenGL context for this slm window. All OpenGL commands now apply to this slm"""
        glfw.make_context_current(self._window)

    def update(self):
        """Sends the new phase pattern to be displayed on the SLM.
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
        glBindFramebuffer(GL_FRAMEBUFFER, self._frame_buffer._frame_buffer)
        glClear(GL_COLOR_BUFFER_BIT)
        for patch in self.patches:
            patch.draw()

        # then draw the frame buffer to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._frame_buffer.draw()

        glfw.poll_events()  # process window messages

        # start 'moving' phase, then display the newly rendered image
        self._start()
        glfw.swap_buffers(self._window)

        # wait for buffer swap to complete (this should be directly after a vsync, so returning from this
        # function _should_ be synced with the vsync)
        glFinish()

        # call _start again to update the _end_time_ns property,
        # since some time has passed waiting for the vsync
        self._start()

    @property
    def latency(self) -> Quantity[u.ms]:
        """The latency (a.k.a. 'idle time')
        represents the time delay between the vertical retrace
        and the start of the SLM response to then new frame.

        The latency can be specified in milliseconds (u.ms) or multiples of the frame period (unitless).
        """
        return (self._latency if isinstance(self._latency, Quantity) else self._latency / self.refresh_rate).to(u.ms)

    @latency.setter
    def latency(self, value: Union[Quantity[u.ms], int]):
        self._latency = value

    @property
    def duration(self) -> Quantity[u.ms]:
        """The duration (a.k.a. 'settle time')
        represents the time between the the start of the SLM response
         and sufficient stabilization of the phase pattern.

        The duration can be specified in milliseconds (u.ms) or multiples of the frame period (unitless).
        """
        return (self._duration if isinstance(self._duration, Quantity) else self._duration / self.refresh_rate).to(u.ms)

    @duration.setter
    def duration(self, value: Union[Quantity[u.ms], int]):
        self._duration = value

    @property
    def transform(self) -> Transform:
        """Global transformation matrix

        The transform determines how the vertex coordinates that make up the shape of a Patch (see
        :class:`Patch`) are mapped to the SLM window.

        When setting a transform, alternatively a string value of 'full', 'short' or 'long' can be used.
        Here, 'full' means that the coordinate range (-1,-1) to (1,1) is mapped to the entire SLM window.
        If the window is not square, this means that the coordinates are anisotropic.

        'short' and 'long' map the coordinate range (-1,-1) to (1,1) to a square.
        'short' means that the square is scaled to fill the short side of the SLM (introducing zero-padding at the
        edges).
        'long' means that the square is scaled to fill the long side of the SLM
        (causing part of the coordinate range to be cropped because these coordinates correspond to points outside
        the SLM window).
        For a square SLM, 'full', 'short' and 'long' are all equivalent.
        In all three cases, (-1,-1) corresponds to the the top-left corner of the screen,
        and (1,-1) to the bottom-left corner.
        This convention is consistent with that used in numpy/matplotlib
        """
        return self._transform

    @transform.setter
    def transform(self, value: Union[Transform, str]):
        width = self._shape[1]
        height = self._shape[0]
        if isinstance(value, Transform):
            self._transform = value
        elif value == 'full':
            self._transform = Transform()
        elif value == 'short' or value == 'long':
            scale_width = (width > height) == (value == 'short')
            if scale_width:
                self._transform = Transform(np.array(((1.0, 0.0), (0.0, height / width))))
            else:
                self._transform = Transform(np.array(((width / height, 0.0), (0.0, 1.0))))
        else:
            raise ValueError("Unsupported transform type")

        # TODO: x/y convention!
        padded = self._transform.opencl_matrix()

        self.activate()  # activate OpenGL context of current SLM window
        glBindBuffer(GL_UNIFORM_BUFFER, self._globals)
        glBufferData(GL_UNIFORM_BUFFER, padded.size * 4, padded, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self._globals)  # connect buffer to binding point 1

    @property
    def lut(self) -> Sequence[int]:
        """Lookup table that is used to map the wrapped phase range of 0-2pi to gray values
        (represented in a range from 0 to 256). By default, this is just range(256).
        Note that the lookup table need not contain 256 elements.
        A typlical scenario is to use something like `slm.lut=range(142)` to map the 0-2pi range
        to only the first 142 gray values of the slm.
        """
        return self._frame_buffer.lookup_table

    @lut.setter
    def lut(self, lut: Sequence[int]):
        self._frame_buffer.lookup_table = lut[:]

    def set_phases(self, values: ArrayLike, update=True):
        self.primary_patch.set_phases(values, update)

    def get_pixels(self, type='phase'):
        """Read back the pixels currently displayed on the SLM."""
        if type == 'gray_value':
            self.activate()
            glReadBuffer(GL_FRONT)
            data = np.empty(self.shape, dtype='uint8')
            glReadPixels(0, 0, self._shape[1], self._shape[0], GL_RED, GL_UNSIGNED_BYTE, data)

            # flip data upside down, because the OpenGL convention is to have the origin at the bottom left
            # but we want it at the top left (like in numpy)
            return data[::-1, :]
        if type == 'phase':
            return self._frame_buffer.get_pixels()

        raise ValueError(f"Unsupported pixel type {type}")
