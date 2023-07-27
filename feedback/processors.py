from base_device_properties import *


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


class SingleRoi(Processor):
    def __init__(self, source, x, y, radius = 0.0):
        super().__init__(source)
        self._x = x
        self._y = y
        self._radius = radius

    def read(self):
        image = super().read()
        return image[self.x, self.y]

    @property
    def data_shape(self):
        return (1,)

    @property
    def x(self) -> int:
        """x-coordinate of center of the ROI"""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> int:
        """y-coordinate of center of the ROI"""
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def radius(self) -> float:
        """radius of the ROI in pixels"""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
