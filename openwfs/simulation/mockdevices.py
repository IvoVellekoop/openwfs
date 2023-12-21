from concurrent.futures import Future

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.ndimage import zoom
import time
from typing import Union, Sequence
from ..processors import CropProcessor
from ..core import Detector, Processor, PhaseSLM, Actuator, get_pixel_size


class Generator(Detector):
    """A detector that returns synthetic data, simulating latency and measurement duration.

    Attributes:
        duration (Quantity[u.ms]): Duration for which the generator simulates data acquisition.
        latency (Quantity[u.ms]): Simulated delay before data acquisition begins.

    Methods:
        uniform_noise: Static method to create a generator with uniform noise.
        gaussian_noise: Static method to create a generator with Gaussian noise.
    """

    def __init__(self, generator, *, duration=0 * u.ms, **kwargs):
        """
        Initializes the Generator class, a subclass of Detector, that generates synthetic data.

        Args:
            generator (callable): A function that takes the data shape as input and returns the generated data.
            duration (Quantity[u.ms]): Duration for which the generator simulates the data acquisition.
            **kwargs: Additional keyword arguments to be passed to the Detector base class.
        """
        super().__init__(**kwargs)
        self._duration = duration
        self._generator = generator

    @property
    def duration(self) -> Quantity[u.ms]:
        return self._duration

    @duration.setter
    def duration(self, value: Quantity[u.ms]):
        self._duration = value

    @property
    def latency(self) -> Quantity[u.ms]:
        return self._latency

    @latency.setter
    def latency(self, value: Quantity[u.ms]):
        self._latency = value.to(u.ns)

    @property
    def pixel_size(self) -> Quantity:
        return super().pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size = value.to(u.ns)

    @property
    def data_shape(self):
        return super().data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._data_shape = value

    def _fetch(self, out: Union[np.ndarray, None]) -> np.ndarray:  # noqa
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
        def generator(shape):
            return np.random.default_rng().uniform(low=low, high=high, size=shape)

        return Generator(*args, generator=generator, **kwargs)

    @staticmethod
    def gaussian_noise(*args, average=0.0, standard_deviation=1.0, **kwargs):
        def generator(shape):
            return np.random.default_rng().normal(loc=average, scale=standard_deviation, size=shape)

        return Generator(*args, generator=generator, **kwargs)


class MockSource(Generator):
    """
    Detector that returns pre-set data. Also simulates latency and measurement duration.

    Attributes:
        data (np.ndarray): Pre-set data to be returned by the mock source.
        pixel_size (Quantity, optional): Size of each pixel in the data.
        extent (Union[Quantity, float, Sequence[float], None], optional): Physical extent of the data array.
    """

    def __init__(self, data, pixel_size: Quantity = None, extent: Union[Quantity, float, Sequence[float], None] = None,
                 **kwargs):
        """
        Initializes the MockSource class, a subclass of Generator, that returns pre-set data.

        Args:
            data (np.ndarray): The pre-set data to be returned by the mock source.
            pixel_size (Quantity, optional): The size of each pixel in the data.
            extent (Union[Quantity, float, Sequence[float], None], optional): The physical extent of the data array.
            **kwargs: Additional keyword arguments.
        """
        def generator(data_shape):
            assert data_shape == self._data.shape
            return self._data.copy()

        if pixel_size is None and extent is not None:
            pixel_size = Quantity(extent) / data.shape

        super().__init__(generator=generator, data_shape=data.shape,
                         pixel_size=pixel_size if pixel_size is not None else get_pixel_size(data),
                         **kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._data_shape = value.shape

    @property
    def data_shape(self):
        return super().data_shape


class ADCProcessor(Processor):
    """Mimics an analog-digital converter.

    At the moment, only positive input and output values are supported.

    Attributes:
        analog_max(float or None): maximum value that the ADC can handle as input,
            this value and all higher values are converted to `digital_max`.
            When set to 0.0, the input signal is scaled automatically so that the maximum corresponds to
            `digital_max`
        digital_max(int): maximum value that the ADC can output.
            Default value is 0xFFFF (16 bits)
        shot_noise(bool): when True, apply Poisson noise to the data instead of rounding
    """

    def __init__(self, source: Detector, analog_max: float = 0.0, digital_max: int = 0xFFFF,
                 shot_noise: bool = False):
        """
        Initializes the ADCProcessor class, which mimics an analog-digital converter.

        Args:
            source (Detector): The source detector providing analog data.
            analog_max (float): The maximum analog value that can be handled by the ADC. If set to 0, the maximum
            value will be set to the maximum value in the data passed by source. Note that this means 2 measurements
            are no longer proportional to each other, if they would be in the source.

            digital_max (int): The maximum digital value that the ADC can output, default is unsigned 16-bit maximum.
            shot_noise (bool): Flag to determine if Poisson noise should be applied instead of rounding.
        """
        super().__init__(source)
        self._analog_max = None
        self._digital_max = None
        self._shot_noise = None
        self.shot_noise = shot_noise
        self.analog_max = analog_max  # check value
        self.digital_max = digital_max  # check value

    def _fetch(self, out: Union[np.ndarray, None], data) -> np.ndarray:  # noqa
        """Clips the data to the range of the ADC, and digitizes the values."""
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
    def analog_max(self) -> float:
        return self._analog_max

    @analog_max.setter
    def analog_max(self, value):
        if value < 0.0:
            raise ValueError('analog_max cannot be negative')
        self._analog_max = value

    @property
    def digital_max(self) -> int:
        return self._digital_max

    @digital_max.setter
    def digital_max(self, value):
        if value < 0 or value > 0xFFFF:
            raise ValueError('digital_max must be between 0 and 0xFFFF')
        self._digital_max = int(value)

    @property
    def shot_noise(self) -> bool:
        return self._shot_noise

    @shot_noise.setter
    def shot_noise(self, value: bool):
        self._shot_noise = value


class MockCamera(ADCProcessor):
    """Wraps any 2-d image source as a camera.

    To implement the camera interface (see bootstrap.py), in addition to the Detector interface,
    we must implement the following:

    Attributes:
        left (int): Left position of the ROI in the source data.
        top (int): Top position of the ROI in the source data.
        height (int): Height of the ROI.
        width (int): Width of the ROI.
        data_shape (tuple): Shape of the image data to be captured.

    In addition, the data should be returned as uint16.
    Conversion to uint16 is implemented in the ADCProcessor base class.
    """

    def __init__(self, source: Detector, shape: Union[Sequence[int], None] = None,
                 pos: Union[Sequence[int], None] = None, **kwargs):
        """
        Args:
            source (Detector): The source detector to be wrapped.
            shape (Union[Sequence[int], None], optional): The shape of the image data to be captured.
            pos (Union[Sequence[int], None], optional): The position on the source from where the image is captured.
            **kwargs: Additional keyword arguments to be passed to the Detector base class.
        """
        self._crop = CropProcessor(source, shape=shape, pos=pos)
        super().__init__(source=self._crop, **kwargs)

    @property
    def left(self) -> int:
        return self._crop.pos[1]

    @left.setter
    def left(self, value: int):
        self._crop.pos = (self._crop.pos[0], value)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def top(self) -> int:
        return self._crop.pos[0]

    @top.setter
    def top(self, value: int):
        self._crop.pos = (value, self._crop.pos[1])

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value: int):
        self.data_shape = (value, self.data_shape[1])

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value: int):
        self.data_shape = (self.data_shape[0], value)

    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._crop.data_shape = value
        self._data_shape = value

    @property
    def duration(self) -> Quantity[u.ms]:
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value.to(u.ms)


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


class MockSLM(PhaseSLM):
    """
    A mock version of a phase-only spatial light modulator.

    Attributes:
        phases (np.ndarray): Current phase pattern on the SLM.

    Methods:
        set_phases(values: Union[np.ndarray, float], update=True): Set the phase pattern on the SLM.
        pixels() -> Detector: Returns a `camera` that mimics the current phase on the SLM.
    """
    def __init__(self, shape):
        """

        Args:
            shape (Sequence[int]): The 2D shape of the SLM.
        """
        super().__init__()
        if len(shape) != 2:
            raise ValueError("Shape of the SLM should be 2-dimensional.")

        self._back_buffer = np.zeros(shape, 'float32')
        self._monitor = MockSource(self._back_buffer, pixel_size=2.0 / np.min(shape) * u.dimensionless_unscaled)

    def update(self):
        self._start()  # wait for detectors to finish
        self._monitor.data = self._back_buffer.copy()
        self._back_buffer[:] = 0.0

    def set_phases(self, values: Union[np.ndarray, float], update=True):
        """
        Set the phase pattern on the SLM. The values are scaled to fit the SLM's shape.

        This method sets a new phase pattern on the SLM. If the input values have a different shape than the SLM,
        they are scaled to fit.

        Args:
            values (Union[np.ndarray, float]): The new phase values to be set on the SLM. This can be a 2D array of
                values or a single float value. If it's a 2D array, it should represent the phase pattern to be
                displayed on the SLM. If it's a single float value, this value is broadcasted over the entire SLM.
            update (bool, optional): If True, the SLM is updated with the new values immediately. Defaults to True.
        """
        values = np.atleast_2d(values)
        scale = np.array(self._back_buffer.shape) / np.array(values.shape)
        # TODO: replace by cv2, with area interpolation
        zoom(values, scale, output=self._back_buffer, order=0)
        if update:
            self.update()

    @property
    def phases(self):
        return self._monitor.data

    def pixels(self) -> Detector:
        """Returns a `camera` that returns the current phase on the SLM.

        The camera coordinates are spanning the [-1,1] range by default."""
        return self._monitor
