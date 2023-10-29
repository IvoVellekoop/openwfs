import numpy as np
import astropy.units as u
from astropy.units import Quantity
from openwfs.feedback import CropProcessor
from ..core import DataSource


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


class MockCamera(CropProcessor):
    """Wraps any 2-d image source as a camera. The Camera object simply crops the image from the source and converts
    the image to 16 bits."""

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
