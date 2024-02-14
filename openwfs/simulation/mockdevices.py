from collections import deque

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from numpy.typing import ArrayLike
import time
from typing import Sequence, Optional
from ..processors import CropProcessor
from ..core import Detector, Processor, PhaseSLM, Actuator
from ..utilities import ExtentType, get_pixel_size, project, set_extent, get_extent, unitless


class Generator(Detector):
    """A detector that returns synthetic data, simulating latency and measurement duration.

    Methods:
        uniform_noise: Static method to create a generator with uniform noise.
        gaussian_noise: Static method to create a generator with Gaussian noise.
    """

    def __init__(self, generator, data_shape: Sequence[int], pixel_size: Quantity, latency=0 * u.ms,
                 duration=0 * u.ms):
        """
        Initializes the Generator class, a subclass of Detector, that generates synthetic data.

        Args:
            generator (callable): A function that takes the data shape as input and returns the generated data.
            data_shape (tuple): The shape of the data to be generated, may have any number of dimensions.
            pixel_size (Quantity, optional): The size of each pixel in the data.
            duration (Quantity[u.ms]): Duration for which the generator simulates the data acquisition.
            latency (Quantity[u.ms]): Simulated delay before data acquisition begins.
        """
        super().__init__()
        self._latency = latency
        self._pixel_size = pixel_size
        self._data_shape = data_shape
        self._duration = duration
        self._generator = generator

    @property
    def duration(self) -> Quantity[u.ms]:
        """The duration for which the generator simulates the data acquisition."""
        return self._duration

    @duration.setter
    def duration(self, value: Quantity[u.ms]):
        self._duration = value

    @property
    def latency(self) -> Quantity[u.ms]:
        """The simulated delay before data acquisition begins."""
        return self._latency

    @latency.setter
    def latency(self, value: Quantity[u.ms]):
        self._latency = value.to(u.ms)

    @property
    def pixel_size(self) -> Quantity:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size = Quantity(value)

    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._data_shape = value

    def _fetch(self, out: Optional[np.ndarray]) -> np.ndarray:  # noqa
        latency_s = self.latency.to_value(u.s)
        if latency_s > 0.0:
            time.sleep(latency_s)

        if out is None:
            out = self._generator(self.data_shape)
        else:
            out[...] = self._generator(self.data_shape)

        duration_s = self.duration.to_value(u.s)
        if duration_s > 0.0:
            time.sleep(duration_s)
        return out

    @staticmethod
    def uniform_noise(*args, low=0.0, high=1.0, **kwargs):
        """Static method to create a generator that produces uniform noise."""

        def generator(shape):
            return np.random.default_rng().uniform(low=low, high=high, size=shape)

        return Generator(*args, generator=generator, **kwargs)

    @staticmethod
    def gaussian_noise(*args, average=0.0, standard_deviation=1.0, **kwargs):
        """Static method to create a generator that produces Gaussian noise."""

        def generator(shape):
            return np.random.default_rng().normal(loc=average, scale=standard_deviation, size=shape)

        return Generator(*args, generator=generator, **kwargs)


class MockSource(Generator):
    """
    Detector that returns pre-set data. Also simulates latency and measurement duration.
    """

    def __init__(self, data: np.ndarray, pixel_size: Optional[Quantity] = None, extent: Optional[ExtentType] = None,
                 **kwargs):
        """
        Initializes the MockSource

        Args:
            data (np.ndarray): The pre-set data to be returned by the mock source.
            pixel_size (Quantity, optional): The size of each pixel in the data.
                If not specified, the pixel size is calculated from the extent and the data shape.
                If neither pixel size nor extent are specified, the pixel size from `data` is used, if available.
                Otherwise, pixel_size is set to None.
            extent (Optional[ExtentType]): The physical extent of the data array.
                Only used when the pixel size is not specified explicitly.
            **kwargs: Additional keyword arguments to pass to the Generator base class.
        """

        def generator(data_shape):
            assert data_shape == self._data.shape
            return self._data.copy()

        if pixel_size is None:
            if extent is not None:
                pixel_size = Quantity(extent) / data.shape
            else:
                pixel_size = get_pixel_size(data)

        if pixel_size is not None and (np.isscalar(pixel_size) or pixel_size.size == 1) and data.ndim > 1:
            pixel_size = pixel_size.repeat(data.ndim)

        super().__init__(generator=generator, data_shape=data.shape, pixel_size=pixel_size, **kwargs)
        self._data = data

    @property
    def data(self):
        """
        The pre-set data to be returned by the mock source.

        Note:
            When setting the `data` property, the `data_shape` attribute is updated accordingly.
            If the `data` property is set with an array that has the `pixel_size` metadata set,
            the `pixel_size` attribute is also updated accordingly.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        pixel_size = get_pixel_size(value)
        if pixel_size is not None:
            self.pixel_size = pixel_size

    @property
    def data_shape(self):
        return self._data.shape


class ADCProcessor(Processor):
    """Mimics an analog-digital converter.

    At the moment, only positive input and output values are supported.
    """

    def __init__(self, source: Detector, analog_max: float = 0.0, digital_max: int = 0xFFFF,
                 shot_noise: bool = False, gaussian_noise_std: float = 0.0):
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
        super().__init__(source)
        self._analog_max = None
        self._digital_max = None
        self._shot_noise = None
        self._gaussian_noise_std = None
        self.gaussian_noise_std = gaussian_noise_std
        self.shot_noise = shot_noise
        self.analog_max = analog_max  # check value
        self.digital_max = digital_max  # check value
        self._signal_multiplier = 1  # Signal multiplier. Can e.g. be used to turn off the virtual laser

    def _fetch(self, out: Optional[np.ndarray], data) -> np.ndarray:  # noqa
        """Clips the data to the range of the ADC, and digitizes the values."""
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
            if out is None:
                out = np.random.poisson(data)
            else:
                out[...] = np.random.poisson(data)
        else:
            if out is None:
                out = np.rint(data).astype('uint16')
            else:
                out[...] = np.rint(data).astype('uint16')

        return out

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
        self._signal_multiplier = 0

    def laser_unblock(self):
        self._signal_multiplier = 1


class MockCamera(ADCProcessor):
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


class MockXYStage(Actuator):
    """
    Mimics an XY stage actuator
    """

    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        """

        Args:
            step_size_x (Quantity[u.um]): The step size in the x-direction.
            step_size_y (Quantity[u.um]): The step size in the y-direction.
        """
        super().__init__()
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

    @property
    def duration(self) -> Quantity[u.ms]:
        return 0.0 * u.ms


class _MockSLMHardware(Generator):
    def __init__(self,
                 shape: Sequence[int],
                 update_latency: Quantity[u.ms] = 0.0 * u.ms,
                 update_duration: Quantity[u.ms] = 0.0 * u.ms):

        if len(shape) != 2:
            raise ValueError("Shape of the SLM should be 2-dimensional.")
        super().__init__(self._generate, data_shape=shape, pixel_size=Quantity(2.0 / np.min(shape)))
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
        self.phase_response = None

    def _generate(self, shape):
        grey_values = self._update(include_current_time=True)

        # Apply phase_response and return actual phases
        if self.phase_response is None:
            return grey_values * (2 * np.pi / 256)
        else:
            return self.phase_response[np.rint(grey_values).astype(np.uint8)]

    def _update(self, include_current_time):
        """Computes the currently displayed voltage image, based on the queue of set points and their timestamps.

        This function takes the frames from the queue. The ones that have a time stamp corresponding to
        the current time or earlier are merged into the current _state. The rest is kept.
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
        if include_current_time:
            self._step_to_time(current_time)

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
        self._queue.append((phase_image, display_time))
        self._update(False)


class MockSLM(PhaseSLM, Actuator):
    """
    A mock version of a phase-only spatial light modulator.

    This object can simulate many properties of a real SLM, including the phase response function and the lookup table.
    It mimics the timing behavior of a real SLM, including the latency (idle time) and update duration (settle time).
    """

    def __init__(self,
                 shape: Sequence[int],
                 latency: Quantity[u.ms] = 0.0 * u.ms,
                 duration: Quantity[u.ms] = 0.0 * u.ms,
                 update_latency: Quantity[u.ms] = 0.0 * u.ms,
                 update_duration: Quantity[u.ms] = 0.0 * u.ms,
                 refresh_rate: Quantity[u.Hz] = 0 * u.Hz):
        """
        Args:
            shape: The shape (height, width) of the SLM in pixels
            latency: The latency that the OpenWFS framework uses for synchronization.
            duration: The duration that the OpenWFS framework uses for synchronization.
            update_latency: The latency of the simulated SLM. 
                Choose a value different than `latency` to simulate incorrect timing.
            update_duration: The duration of the simulated SLM.
                Choose a value different than `duration` to simulate incorrect timing.
            refresh_rate: Simulated refresh rate. Affects the timing of the `update` method,
                since this will wait until the next vertical retrace. Keep at 0 to disable this feature.
            """
        super().__init__()
        self.refresh_rate = refresh_rate
        self._hardware = _MockSLMHardware(shape, update_latency, update_duration)  # Actual phase
        self._lookup_table = None  # index = input phase (scaled to -> [0, 255]), value = grey value
        self._first_update_ns = time.time_ns()
        self._back_buffer = np.zeros(shape, dtype=np.float32)

    def update(self):
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
            grey_values = np.floor(256 * tx).astype(np.uint8)  # TODO: remove np.floor here and below
        else:
            lookup_index = np.floor(self._lookup_table.shape[0] * tx).astype(np.uint8)  # index into lookup table
            grey_values = self._lookup_table[lookup_index]

        self._hardware.send(grey_values)

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
    def phase_response(self) -> Sequence[float]:
        """
        phase_response function: The phase response of the SLM.
        """
        return self._hardware.phase_response

    @phase_response.setter
    def phase_response(self, value: Sequence[float]):
        self._hardware.phase_response = value

    def set_phases(self, values: ArrayLike, update=True):
        # no docstring, use documentation from base class

        project(np.atleast_2d(values).astype('float32'), out=self._back_buffer, source_extent=(2.0, 2.0),
                out_extent=(2.0, 2.0))
        if update:
            self.update()

    @property
    def phases(self) -> np.ndarray:
        """Current phase pattern on the SLM."""
        return self._hardware.read()

    def pixels(self) -> Detector:
        """Returns a `camera` that returns the current phase on the SLM.

        The camera coordinates are spanning the [-1,1] range by default."""
        return self._hardware

    @property
    def duration(self) -> Quantity[u.ms]:
        return 0.0 * u.ms
