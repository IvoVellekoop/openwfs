import numpy as np
import astropy.units as u
from astropy.units import Quantity


# todo: put somewhere else (in processors.py?)
class Processor:
    def __init__(self, source):
        self.source = source

    def trigger(self):
        self.source.trigger()

    @property
    def data_shape(self):
        return self.source.data_shape

    @property
    def measurement_time(self):
        return self.source.measurement_time

    @property
    def pixel_size(self):
        return self.source.pixel_size

    def read(self):
        return self.source.read()


class MockImageSource:
    """'Camera' that dynamically generates images using an 'on_trigger' callback.
    Note that this object does not implement the full camera interface. Does not support resizing yet"""

    def __init__(self, on_trigger, data_shape, pixel_size: Quantity[u.um]):
        self._on_trigger = on_trigger
        self._image = np.empty(data_shape, dtype="float32")
        self._measurement_time = 0  # property may be set to something else to mimic acquisition time
        self._triggered = 0  # currently only support a buffer with a single frame
        self.pixel_size = pixel_size.to(u.um)
        self.data_shape = data_shape
        self.measurement_time = 0.0 * u.ms

    def trigger(self):
        if self._triggered != 0:
            raise RuntimeError('Buffer overflow: this camera can only store 1 frame, and the previous frame was not '
                               'read yet')
        self._image = self._on_trigger(self._image)
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
        return MockImageSource(lambda i: image, image.shape, pixel_size.to(u.um))


class CropProcessor(Processor):
    def __init__(self, source, width=None, height=None, left=0, top=0):
        self.source = source
        self.left = left
        self.top = top
        if width is None:
            width = source.data_shape[1]
        if height is None:
            height = source.data_shape[0]
        self._data_shape = None
        self._set_shape(width, height)

    def _set_shape(self, width, height):
        ss = self.source.data_shape
        self._data_shape = (np.minimum(ss[0] - self.top, height), np.minimum(ss[1] - self.left, width))

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._set_shape(value, self.height)

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._set_shape(self.width, value)

    def read(self):
        image = super().read()
        bottom = self.top + self.height
        right = self.left + self.width
        return image[self.top:bottom, self.left:right]


class MockCamera(CropProcessor):
    """Wraps any 2-d image source as a camera. The Camera object simply crops the image from the source"""
    def __init__(self, source, saturation=1000, width=None, height=None, left=0, top=0):
        super().__init__(source, width, height, left, top)
        self._saturation = saturation

    def read(self):
        im = super().read() * (0xFFFF / self.saturation)
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
        self._position_x = 0.0 * u.um
        self._position_y = 0.0 * u.um

    @property
    def position_x(self) -> Quantity[u.um]:
        return self._position_x

    @position_x.setter
    def position_x(self, value: Quantity[u.um]):
        self._position_x = value.to(u.um)

    @property
    def position_y(self) -> Quantity[u.um]:
        return self._position_y

    @position_y.setter
    def position_y(self, value: Quantity[u.um]):
        self._position_y = value.to(u.um)
