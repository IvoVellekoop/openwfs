import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.ndimage import zoom
import time
from typing import Union
from ..feedback import CropProcessor
from ..core import DataSource, Processor, PhaseSLM, get_pixel_size


class Generator(DataSource):
    """Detector that returns synthetic data.
    Also simulates latency and measurement duration.
    """

    def __init__(self, generator, *, duration=0 * u.ms, **kwargs):
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
    """Detector that static data.
    Also simulates latency and measurement duration.
    """

    def __init__(self, data, pixel_size: Quantity[u.um] = None, **kwargs):
        def generator(data_shape):
            assert data_shape == self._data.shape
            return self._data

        super().__init__(generator=generator, data_shape=data.shape,
                         pixel_size=pixel_size if pixel_size is not None else get_pixel_size(data),
                         **kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.wait()
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

    def __init__(self, source: DataSource, analog_max: float = 0.0, digital_max: int = 0xFFFF,
                 shot_noise: bool = False):
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
            data = data * (self.digital_max / np.max(data))
        else:
            data = np.clip(data * (self.digital_max / self.analog_max), 0, self.digital_max)

        if self._shot_noise:
            if out is None:
                out = np.random.poisson(data)
            else:
                out[...] = np.random.poisson(data)
        else:
            return np.rint(data, out=out, dtype='uint16')

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

    To implement the camera interface (see bootstrap.py), in addition to the DataSource interface,
    we must implement the following functions:
        top: int
        left: int
        height: int
        width: int

    These properties are forwarded to a CropProcessor held internally.

    In addition, the data should be returned as uint16.
    Conversion to uint16 is implemented in the ADCProcessor base class.
    """

    def __init__(self, source: DataSource, width: int = None, height: int = None, left: int = 0, top: int = 0,
                 analog_max: float = 0.0, digital_max: int = 0xFFF):
        self._cropped = CropProcessor(source, width, height, left, top)
        super().__init__(source=self._cropped, digital_max=digital_max, analog_max=analog_max)

    @property
    def left(self):
        return self._cropped.left

    @left.setter
    def left(self, value):
        self._cropped.left = value

    @property
    def right(self):
        return self._cropped.right

    @right.setter
    def right(self, value):
        self._cropped.right = value

    @property
    def top(self):
        return self._cropped.top

    @top.setter
    def top(self, value):
        self._cropped.top = value

    @property
    def bottom(self):
        return self._cropped.bottom

    @bottom.setter
    def bottom(self, value):
        self._cropped.bottom = value


class MockXYStage:
    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
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

    def wait(self):
        pass


class MockSLM(PhaseSLM):
    def __init__(self, width: int, height: int):
        super().__init__()
        self._back_buffer = np.zeros((height, width), 'float32')
        self._front_buffer = np.zeros((height, width), 'float32')
        self._monitor = MockSource(self._front_buffer, pixel_size=1.0 / min(width, height) * u.dimensionless_unscaled)

    def update(self):
        np.copyto(self._front_buffer, self._back_buffer)
        self._back_buffer[:] = 0.0

    def set_phases(self, values: Union[np.ndarray, float], update=True):
        values = np.atleast_2d(values)
        scale = np.array(self._back_buffer.shape) / np.array(values.shape)
        zoom(values, scale, output=self._back_buffer, order=0)
        if update:
            self.update()

    def pixels(self) -> DataSource:
        return self._monitor
