import numpy as np
import astropy.units as u
from astropy.units import Quantity

class MockImageSource:
    """'Camera' that dynamically generates images using an 'on_trigger' callback.
    Note that this object does not implement the full camera interface. Does not support resizing yet"""

    @staticmethod
    def from_image(image, pixel_size: Quantity[u.um]):
        """Create a MockImageSource that displays the given numpy array as image.
        :param image : 2-d numpy array to return on 'read'
        :param pixel_size : size of a single pixel, must have a pint unit of type length
        """
        return MockImageSource(lambda i: image, image.shape, pixel_size.to(u.um))

    def __init__(self, on_trigger, data_shape, pixel_size: Quantity[u.um]):
        self._on_trigger = on_trigger
        self._image = np.empty(data_shape, dtype="float32")
        self._measurement_time = 0  # property may be set to something else to mimic acquisition time
        self._triggered = 0  # currently only support a buffer with a single frame
        self.pixel_size = pixel_size.to(u.um)
        self.data_shape = data_shape
        self.measurement_time = 0.0

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


class MockXYStage:
    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        self.step_size_x = Quantity(step_size_x, u.um)
        self.step_size_y = Quantity(step_size_y, u.um)
        self._position_x = 0.0 * u.um
        self._position_y = 0.0 * u.um

    @property
    def position_x(self) -> Quantity[u.um]:
        return self._position_x

    @position_x.setter
    def position_x(self, value: Quantity[u.um]):
        self._position_x = Quantity(value, u.um)

    @property
    def position_y(self) -> Quantity[u.um]:
        return self._position_y

    @position_y.setter
    def position_y(self, value: Quantity[u.um]):
        self._position_y = Quantity(value, u.um)
