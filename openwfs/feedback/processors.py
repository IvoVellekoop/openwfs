import numpy as np


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
    def __init__(self, source, x, y, radius=0.0):
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


class CropProcessor(Processor):
    def __init__(self, source, width=None, height=None, left=0, top=0):
        super().__init__(source)
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
