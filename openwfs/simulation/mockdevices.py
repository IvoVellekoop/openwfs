import time
from collections import deque
from typing import Sequence, Optional, Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike

from ..core import Detector, Processor, PhaseSLM, Actuator
from ..processors import CropProcessor
from ..utilities import ExtentType, get_pixel_size, project, unitless


class StaticSource(Detector):
    """
    Detector that returns pre-set data. Also simulates latency and measurement duration.
    """

    def __init__(self, data: np.ndarray, pixel_size: Optional[Quantity] = None, extent: Optional[ExtentType] = None,
                 latency: Quantity[u.ms] = 0 * u.ms, duration: Quantity[u.ms] = 0 * u.ms, multi_threaded: bool = None):
        """
        Initializes the MockSource
        TODO: factor out the latency and duration into a separate class?
        Args:
            data (np.ndarray): The pre-set data to be returned by the mock source.
            pixel_size (Quantity, optional): The size of each pixel in the data.
                If not specified, the pixel size is calculated from the extent and the data shape.
                If neither pixel size nor extent are specified, the pixel size from `data` is used, if available.
                Otherwise, pixel_size is set to None.
            extent (Optional[ExtentType]): The physical extent of the data array.
                Only used when the pixel size is not specified explicitly.
        """
        if pixel_size is None:
            if extent is not None:
                pixel_size = Quantity(extent) / data.shape
            else:
                pixel_size = get_pixel_size(data)

        if pixel_size is not None and (np.isscalar(pixel_size) or pixel_size.size == 1) and data.ndim > 1:
            pixel_size = pixel_size.repeat(data.ndim)

        if multi_threaded is None:
            multi_threaded = latency > 0 * u.ms or duration > 0 * u.ms

        self._data = data
        super().__init__(data_shape=data.shape, pixel_size=pixel_size, latency=latency, duration=duration,
                         multi_threaded=multi_threaded)

    def _fetch(self) -> np.ndarray:  # noqa
        total_time_s = self.latency.to_value(u.s) + self.duration.to_value(u.s)
        if total_time_s > 0.0:
            time.sleep(total_time_s)
        return self._data

    @property
    def data(self):
        """
        The pre-set data to be returned by the mock source.

        Note:
            The data is not copied.
            When setting the `data` property, the `data_shape` attribute is updated accordingly.
            If the `data` property is set with an array that has the `pixel_size` metadata set,
            the `pixel_size` attribute is also updated accordingly.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._pixel_size = get_pixel_size(value)
        self._data_shape = value.shape


class NoiseSource(Detector):
    def __init__(self, noise_type: str, *, data_shape: tuple[int, ...], pixel_size: Quantity, multi_threaded=True,
                 **kwargs):
        self._noise_type = noise_type
        self._noise_arguments = kwargs
        self._rng = np.random.default_rng()
        super().__init__(data_shape=data_shape, pixel_size=pixel_size, latency=0 * u.ms, duration=0 * u.ms,
                         multi_threaded=multi_threaded)

    def _fetch(self) -> np.ndarray:  # noqa
        if self._noise_type == 'uniform':
            return self._rng.uniform(**self._noise_arguments, size=self.data_shape)
        elif self._noise_type == 'gaussian':
            return self._rng.normal(**self._noise_arguments, size=self.data_shape)
        else:
            raise ValueError(f'Unknown noise type: {self._noise_type}')

    @Detector.data_shape.setter
    def data_shape(self, value):
        self._data_shape = tuple(value)


class ADCProcessor(Processor):
    """Mimics an analog-digital converter.

    At the moment, only positive input and output values are supported.
    """

    def __init__(self, source: Detector, analog_max: float = 0.0, digital_max: int = 0xFFFF,
                 shot_noise: bool = False, gaussian_noise_std: float = 0.0, multi_threaded: bool = True):
        """
        Initializes the ADCProcessor class, which mimics an analog-digital converter.

        Args:
            source (Detector): The source detector providing analog data.
            analog_max (float): The maximum analog value that can be handled by the ADC.
                If set to 0.0, each measurement will be automatically scaled so that the maximum
                value in the data set returned by `source` is converted to `digital_max`.
                Note that this means that the values from two different measurements cannot be compared quantitatively.

            digital_max (int):
                The maximum digital value that the ADC can output, default is unsigned 16-bit maximum.

            shot_noise (bool):
                Flag to determine if Poisson noise should be applied instead of rounding.
                Useful for realistically simulating detectors.

            gaussian_noise_std (float):
                If >0, add gaussian noise with std of this value to the data.
        """
        super().__init__(source, multi_threaded=multi_threaded)
        self._analog_max = None
        self._digital_max = None
        self._shot_noise = None
        self._gaussian_noise_std = None
        self.gaussian_noise_std = gaussian_noise_std
        self.shot_noise = shot_noise
        self.analog_max = analog_max  # check value
        self.digital_max = digital_max  # check value
        self._signal_multiplier = 1  # Signal multiplier. Can e.g. be used to turn off the virtual laser

    def _fetch(self, data) -> np.ndarray:  # noqa
        """Clips the data to the range of the ADC, and digitizes the values."""
        # TODO: gaussian noise should be added after shot noise
        # Add gaussian noise if requested
        if self._gaussian_noise_std > 0.0:
            dtype = data.dtype
            rng = np.random.default_rng()
            gaussian_noise = self._gaussian_noise_std * rng.standard_normal(size=data.shape)
            data = (self._signal_multiplier * data.astype('float64') + gaussian_noise).clip(0, self.digital_max).astype(
                dtype)

        if self.analog_max == 0.0:
            max = np.max(data)
            if max > 0.0:
                data = data * (self.digital_max / np.max(data))  # auto-scale to maximum value
        else:
            data = np.clip(data * (self.digital_max / self.analog_max), 0, self.digital_max)

        if self._shot_noise:
            return np.random.poisson(data)
        else:
            return np.rint(data).astype('uint16')

    @property
    def analog_max(self) -> Optional[float]:
        """Maximum value that the ADC can handle as input

        This value and all higher values are converted to `digital_max`.
        When set to 0.0, the input signal is scaled automatically so that the maximum corresponds to
        `digital_max`
        """
        return self._analog_max

    @analog_max.setter
    def analog_max(self, value):
        if value < 0.0:
            raise ValueError('analog_max cannot be negative')
        self._analog_max = value

    @property
    def digital_max(self) -> int:
        """Maximum value that the ADC can output.

        Default value is 0xFFFF (16 bits)
        """
        return self._digital_max

    @digital_max.setter
    def digital_max(self, value):
        if value < 0 or value > 0xFFFF:
            raise ValueError('digital_max must be between 0 and 0xFFFF')
        self._digital_max = int(value)

    @property
    def shot_noise(self) -> bool:
        """when True, apply Poisson noise to the data instead of rounding"""
        return self._shot_noise

    @shot_noise.setter
    def shot_noise(self, value: bool):
        self._shot_noise = value

    @property
    def gaussian_noise_std(self) -> int:
        return self._gaussian_noise_std

    @gaussian_noise_std.setter
    def gaussian_noise_std(self, value: int):
        self._gaussian_noise_std = value

    def laser_block(self):
        """Simulate blocking of the light source by setting signal multiplier to 0."""
        self._signal_multiplier = 0

    def laser_unblock(self):
        """Simulate blocking of the light source by setting signal multiplier to 1."""
        self._signal_multiplier = 1


class Camera(ADCProcessor):
    """Wraps any 2-d image source as a camera.

    To implement the camera interface, in addition to the Detector interface,
    we must implement left,right,top, and bottom.
    In addition, the data should be returned as uint16.
    Conversion to uint16 is implemented in the ADCProcessor base class.
    """

    def __init__(self, source: Detector, shape: Optional[Sequence[int]] = None,
                 pos: Optional[Sequence[int]] = None, **kwargs):
        """
        Args:
            source (Detector): The source detector to be wrapped.
            shape (Optional[Sequence[int]]): The shape of the image data to be captured.
            pos (Optional[Sequence[int]]): The position on the source from where the image is captured.
            **kwargs: Additional keyword arguments to be passed to the Detector base class.

            TODO: move left-right-top-bottom to CropProcessor.
            Expose the properties of CropProcessor as properties of MockCamera automatically by copying from __dict__?
        """
        self._crop = CropProcessor(source, shape=shape, pos=pos)
        super().__init__(source=self._crop, **kwargs)

    @property
    def left(self) -> int:
        """left (int): Horizontal start position of the ROI."""
        return self._crop.pos[1]

    @left.setter
    def left(self, value: int):
        self._crop.pos = (self._crop.pos[0], value)

    @property
    def right(self) -> int:
        """right (int): Horizontal end position of the ROI."""
        return self.left + self.width

    @property
    def top(self) -> int:
        """top (int): Vertical start position of the ROI."""
        return self._crop.pos[0]

    @top.setter
    def top(self, value: int):
        self._crop.pos = (value, self._crop.pos[1])

    @property
    def bottom(self) -> int:
        """bottom (int): Vertical end position of the ROI."""
        return self.top + self.height

    @property
    def height(self) -> int:
        """Height of the ROI."""
        return self._crop.data_shape[0]

    @height.setter
    def height(self, value: int):
        self._crop.data_shape = (value, self._crop.data_shape[1])

    @property
    def width(self) -> int:
        """Width of the ROI."""
        return self._crop.data_shape[1]

    @width.setter
    def width(self, value: int):
        self._crop.data_shape = (self._crop.data_shape[0], value)

    @property
    def data_shape(self):
        return self._crop.data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._crop.data_shape = value


class XYStage(Actuator):
    """
    Mimics an XY stage actuator
    """

    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        """

        Args:
            step_size_x (Quantity[u.um]): The step size in the x-direction.
            step_size_y (Quantity[u.um]): The step size in the y-direction.
        """
        super().__init__(duration=0 * u.ms, latency=0 * u.ms)
        self.step_size_x = step_size_x.to(u.um)
        self.step_size_y = step_size_y.to(u.um)
        self._y = 0.0 * u.um
        self._x = 0.0 * u.um

    @property
    def x(self) -> Quantity[u.um]:
        return self._x

    @x.setter
    def x(self, value: Quantity[u.um]):
        self._x = value.to(u.um)

    @property
    def y(self) -> Quantity[u.um]:
        return self._y

    @y.setter
    def y(self, value: Quantity[u.um]):
        self._y = value.to(u.um)

    def home(self):
        self._x = 0.0 * u.um
        self._y = 0.0 * u.um


class _SLMField(Processor):
    """Computes the field reflected by a MockSLM."""

    def __init__(self, slm_phases: Detector, field_amplitude: ArrayLike = 1.0,
                 non_modulated_field_fraction: float = 0.0):
        """
        Args:
            slm_phases: The `Detector` that returns the phases of the slm pixels.
            field_amplitude: Field amplitude of the modulated pixels.
            non_modulated_field_fraction: Non-modulated field (e.g. a front reflection).
        """
        super().__init__(slm_phases, multi_threaded=False)
        self.modulated_field_amplitude = field_amplitude
        self.non_modulated_field = non_modulated_field_fraction

    def _fetch(self, slm_phases: np.ndarray) -> np.ndarray:  # noqa
        """
        Updates the complex field output of the SLM. The output field is the sum of the modulated field and the
        non-modulated field.
        """
        return self.modulated_field_amplitude * (np.exp(1j * slm_phases) + self.non_modulated_field)


class _SLMTiming(Detector):
    """Class to simulate the timing of an SLM.

    This class simulates latency (`update_latency`) and
    stabilization (`update_duration`) of the SLM. It does not simulate
    the refresh rate, or the conversion of gray values to phases.
    """

    def __init__(self,
                 shape: tuple[int],
                 update_latency: Quantity[u.ms] = 0.0 * u.ms,
                 update_duration: Quantity[u.ms] = 0.0 * u.ms):

        if len(shape) != 2:
            raise ValueError("Shape of the SLM should be 2-dimensional.")
        super().__init__(data_shape=shape, pixel_size=Quantity(2.0 / np.min(shape)), latency=0 * u.ms,
                         duration=0 * u.ms, multi_threaded=False)
        self.update_latency = update_latency
        self.update_duration = update_duration

        # _set_point are the voltages that the SLM hardware is currently sending to the display.
        #   The display may not have stabilized yet
        # _state are the voltages currently on the display. They will asymptotically approach _set_point
        # _state_timestamp is the time when the state was last updated
        # _queue is the queue of upcoming frames (set points and their timestamps)
        # that are sent from the PC to the SLM, but not yet displayed.
        self._set_point = np.zeros(shape, dtype=np.float32)
        self._state = np.zeros(shape, dtype=np.float32)
        self._state_timestamp = 0.0
        self._queue = deque()  # Queue to hold phase images and their timestamps

    def _fetch(self):
        return self._update(None)

    def _update(self, append):
        """Computes the currently displayed voltage image, based on the queue of set points and their timestamps.

        This function takes the frames from the queue. The ones that have a time stamp corresponding to
        the current time or earlier are merged into the current _state. The rest is kept.
        Note: if ever converting this object to multi-threading, this function should be protected by a lock.
        """
        current_time = time.time_ns() * u.ns
        while True:
            # peek the first element in the queue
            if not self._queue:
                break
            set_point, timestamp = self._queue[0]
            if timestamp > current_time:
                break

            # we found a frame that is or has been displayed
            # we step the simulation forward to the time of the frame
            self._step_to_time(timestamp)
            self._set_point = set_point
            self._queue.popleft()

        # finally, compute the current state
        if append is None:
            self._step_to_time(current_time)
        else:
            self._queue.append(append)

        return self._state

    def _step_to_time(self, time):
        """Step the simulation forward to a given time."""
        if self.update_duration > 0:
            a = np.exp(-(time - self._state_timestamp) / self.update_duration)
            self._state = a * self._state + (1 - a) * self._set_point
        else:
            self._state = self._set_point
        self._state_timestamp = time

    def send(self, phase_image):
        """Send a phase image to the SLM. This method is called by the SLM object.

        The image is not displayed directly. Instead, it is added to a queue with a timestamp
        attached, and it only takes effect after update_latency seconds have passed.

        All old images in the queue are merged with the new image, using an exponential decay.
        """
        display_time = time.time_ns() * u.ns + self.update_latency
        self._update((phase_image, display_time))


class _SLMPhaseResponse(Processor):
    def __init__(self, source: _SLMTiming, phase_response: Optional[np.ndarray] = None):
        super().__init__(source, multi_threaded=False)
        self.phase_response = phase_response

    def _fetch(self, grey_values: np.ndarray) -> np.ndarray:  # noqa
        if self.phase_response is None:
            return grey_values * (2 * np.pi / 256)
        else:
            return self.phase_response[np.rint(grey_values).astype(np.uint8)]


class SLM(PhaseSLM, Actuator):
    """
    A mock version of a phase-only spatial light modulator. Some properties are available to simulate physical
    phenomena such as imperfect phase response, and front reflections (which cause non-modulated light).
    """
    __slots__ = ('_hardware_fields', '_hardware_phases', '_hardware_timing', '_back_buffer',
                 'refresh_rate', '_first_update_ns', '_lookup_table')

    def __init__(self,
                 shape: Sequence[int],
                 latency: Quantity[u.ms] = 0.0 * u.ms,
                 duration: Quantity[u.ms] = 0.0 * u.ms,
                 update_latency: Quantity[u.ms] = 0.0 * u.ms,
                 update_duration: Quantity[u.ms] = 0.0 * u.ms,
                 refresh_rate: Quantity[u.Hz] = 0 * u.Hz,
                 field_amplitude: Union[np.ndarray, float, None] = 1.0,
                 non_modulated_field_fraction: float = 0.0,
                 phase_response: Optional[np.ndarray] = None,
                 ):
        """

        Args:
            shape: The 2D shape of the SLM.
            field_amplitude: Field amplitude of the modulated light.
            non_modulated_field_fraction (float): fraction of the field that is not modulated,
                typically due to reflection at the front surface of the SLM.
            shape: The shape (height, width) of the SLM in pixels
            latency: The latency that the OpenWFS framework uses for synchronization.
            duration: The duration that the OpenWFS framework uses for synchronization.
            update_latency: The latency of the simulated SLM.
                Choose a value different from `latency` to simulate incorrect timing.
            update_duration: The duration of the simulated SLM.
                Choose a value different from `duration` to simulate incorrect timing.
            refresh_rate: Simulated refresh rate. Affects the timing of the `update` method,
                since this will wait until the next vertical retrace. Keep at 0 to disable this feature.
            """
        super().__init__(latency=latency, duration=duration)
        self.refresh_rate = refresh_rate
        # Simulates transferring frames to the SLM
        self._hardware_timing = _SLMTiming(shape, update_latency, update_duration)
        self._hardware_phases = _SLMPhaseResponse(self._hardware_timing,
                                                  phase_response)  # Simulates reading the phase from the SLM
        self._hardware_fields = _SLMField(self._hardware_phases, field_amplitude,
                                          non_modulated_field_fraction)  # Simulates reading the field from the SLM
        self._lookup_table = None  # index = input phase (scaled to -> [0, 255]), value = grey value
        self._first_update_ns = time.time_ns()
        self._back_buffer = np.zeros(shape, dtype=np.float32)

    def update(self):
        """Sends the current phase image to the simulated SLM hardware."""

        self._start()  # wait for detectors to finish
        if self.refresh_rate > 0:
            # wait for the vertical retrace
            time_in_frames = unitless((time.time_ns() - self._first_update_ns) * u.ns * self.refresh_rate)
            time_to_next_frame = (np.ceil(time_in_frames) - time_in_frames) / self.refresh_rate
            time.sleep(time_to_next_frame.tovalue(u.s))
            # update the start time (this is also done in the actual SLM)
            self._start()

        # Apply lookup table and compute grey values from intended phases
        # This uses the same conversion as in the shader:
        # first compute tx = phi * / (2 * pi) + 0.5 / 256
        # use the fractional part of tx to index the lookup table, where:
        # - tx =0 maps to the start of the first element
        # - tx=1-δ maps to the end of the last element
        tx = self._back_buffer * (1 / (2 * np.pi)) + (0.5 / 256)
        tx = tx - np.floor(tx)  # fractional part of tx
        if self._lookup_table is None:
            grey_values = (256 * tx).astype(np.uint8)
        else:
            lookup_index = (self._lookup_table.shape[0] * tx).astype(np.uint8)  # index into lookup table
            grey_values = self._lookup_table[lookup_index]

        self._hardware_timing.send(grey_values)

    @property
    def lookup_table(self) -> Sequence[int]:
        """Lookup table that is used to map the wrapped phase range of 0-2pi to gray values
        (represented in a range from 0 to 256). By default, this is just range(256).
        Note that the lookup table need not contain 256 elements.
        A typical scenario is to use something like `slm.lookup_table=range(142)` to map the 0-2pi range
        to only the first 142 gray values of the slm.
        """
        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, value: Sequence[int]):
        self._lookup_table = np.asarray(value)

    @property
    def phase_response(self) -> Optional[np.ndarray]:
        """Phase as a function of pixel gray value.

        This lookup table mimics the phase response of the SLM hardware.
        When it is omitted, the phase response is assumed to be linear:
        phase = 2 * pi * grey_value / 256
        """
        return self._hardware_phases.phase_response

    @phase_response.setter
    def phase_response(self, value: Sequence[float]):
        self._hardware_phases.phase_response = value

    def set_phases(self, values: ArrayLike, update=True):
        # no docstring, use documentation from base class

        # Copy the phase image to the back buffer, scaling it as necessary
        project(np.atleast_2d(values).astype('float32'), out=self._back_buffer, source_extent=(2.0, 2.0),
                out_extent=(2.0, 2.0))
        if update:
            self.update()

    def get_monitor(self, monitor_type: str = 'phase') -> Detector:
        """Returns an object to monitor the current state of the SLM.

        The monitor object acts as a camera, that can be used like any camera in the framework.
        Args:
            monitor_type (str): Type of the monitor. May be:
            - 'phase': returns the simulated phase that is currently on the SLM.
               This takes into account the simulated settle time and latency, refresh rate, lookup table
               and phase response.
            - 'field': returns the simulated field that is currently on the SLM.
               Equal to `(exp(1j * phase) + non_modulated_field_fraction) * field_amplitude`
            - 'pixel_value': returns the effective pixel values that are currently displayed on the slm,
               taking into account the simulated settle time and latency, refresh rate and lookup table

        To disable simulating the time-dependent behavior of the SLM,
        leave refresh_rage, update_latency and update_duration at their default values of 0.
        """
        if monitor_type == 'phase':
            return self.phases
        elif monitor_type == 'field':
            return self.field
        elif monitor_type == 'pixel_value':
            return self.pixels
        else:
            raise ValueError(f"Unknown monitor type: {monitor_type}")

    @property
    def pixels(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, these are the effective gray values that are currently displayed on the SLM.
        """
        return self._hardware_timing

    @property
    def field(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, this is the effective field that is currently on the SLM.
        """
        return self._hardware_fields

    @property
    def phases(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, this is the effective phase that is currently on the SLM.
        """
        return self._hardware_phases

    @property
    def duration(self) -> Quantity[u.ms]:
        return 0.0 * u.ms


class Shutter(Processor):
    """
    A mock version of a shutter.
    When open, passes through the input field
    When closed, passes through the input field * 0.0
    """

    def __init__(self, source: Detector):
        super().__init__(source, multi_threaded=False)
        self._open = True

    @property
    def open(self) -> bool:
        return self._open

    @open.setter
    def open(self, value: bool):
        self._open = value

    def _fetch(self, source: np.ndarray) -> np.ndarray:  # noqa
        return source if self._open else 0.0 * source
