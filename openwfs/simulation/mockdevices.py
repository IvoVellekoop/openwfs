import numpy as np
import astropy.units as u
from astropy.units import Quantity
from openwfs.feedback import CropProcessor
from ..core import DataSource, Processor


class MockImageSource(DataSource):
    """'Camera' that dynamically generates images using an 'on_trigger' callback.
    Note that this object does not implement the full camera interface. Does not support resizing yet"""

    def __init__(self, on_trigger, data_shape, pixel_size: Quantity[u.um]):
        self._on_trigger = on_trigger
        self._image = None
        self._measurement_time = 0  # property may be set to something else to mimic acquisition time
        self._triggered = 0  # currently only support a buffer with a single frame
        self._data_shape = data_shape
        self._pixel_size = pixel_size.to(u.um)
        self._measurement_time = 0.0 * u.ms

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def measurement_time(self) -> Quantity:
        return 0.0 * u.s

    @property
    def pixel_size(self) -> Quantity:
        return self._pixel_size

    def trigger(self):
        if self._triggered != 0:
            raise RuntimeError('Buffer overflow: this camera can only store 1 frame, and the previous frame was not '
                               'read yet')
        self._image = self._on_trigger()
        self._triggered += 1

    def read(self):
        if self._triggered == 0:
            raise RuntimeError('Buffer underflow: trying to read an image without triggering the camera')

        self._triggered -= 1
        return self._image

    @staticmethod
    def from_image(image, pixel_size: Quantity[u.um]):
        """Create a MockImageSource that displays the given numpy array as image.
        :param image : 2-d numpy array to return on 'read'
        :param pixel_size : size of a single pixel, must have a pint unit of type length
        """
        return MockImageSource(lambda: image, image.shape, pixel_size)


class ADCProcessor(Processor):
    """Mimics an analog-digital converter.

    At the moment, only positive input and output values are supported.

    Attributes:
        analog_max(float or None): maximum value that the ADC can handle as input,
            this value and all higher values are converted to `digital_max`.
            When set to 0.0, the input signal is scaled automatically so that the maximum corresponds to
            `digital_max`
        digital_max(int): maximum value that the ADC can output.
            defaults to 0xFFFF (16 bits)
        shot_nose(bool): when True, apply Poisson noise to the data instead of rounding
    """

    def __init__(self, source: DataSource, analog_max: float = 0.0, digital_max: int = 0xFFFF,
                 shot_noise: bool = False):
        self._analog_max = analog_max
        self._digital_max = digital_max
        self._shot_noise = shot_noise
        self.analog_max = analog_max  # check value
        self.digital_max = digital_max  # check value
        super().__init__(source)

    def read(self):
        data = super().read()
        if self.analog_max == 0.0:
            data = data * (self.digital_max / np.max(data))
        else:
            data = np.clip(data * (self.digital_max / self.analog_max), 0, self.digital_max)

        if self._shot_noise:
            data = np.random.poisson(data)
        else:
            data = np.round(data)
        return np.array(data, dtype='uint16')

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
    def shot_noise(self, value):
        self._shot_noise = value


class MockCamera(ADCProcessor):
    """Wraps any 2-d image source as a camera.

    To implement the camera interface (see bootstrap.py), in addition to the DataSource interface
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

    @top.setter
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


class NoiseCamera:
    """Fake Camera object that just produces noise."""

    def __init__(self, source, saturation=1000, width=None, height=None, left=0, top=0):
        super().__init__(source, width, height, left, top)
        self._saturation = saturation

    def read(self):
        im = np.clip(super().read() * (0xFFFF / self.saturation), 0, 0xFFFF)
        return np.array(im, dtype='uint16')

    @property
    def saturation(self) -> float:
        return self._saturation

    @saturation.setter
    def saturation(self, value):
        self._saturation = value
