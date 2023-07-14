import numpy as np
from base_device_properties import *

class Processor:
    def __init__(self, source, **kwargs):
        parse_options(self, kwargs)
        self.source = source

    def trigger(self):
        self.source.trigger()

    @property
    def data_shape(self):
        return self.source.data_shape

    @property
    def measurement_time(self):
        return self.source.measurement_time


class SingleRoi(Processor):
    def read(self):
        image = self.source.read()
        return image[self.x, self.y]

    @property
    def data_shape(self):
        return (1,)

    x = int_property(doc="x-coordinate of center of the ROI")
    y = int_property(doc="y-coordinate of center of the ROI")
    radius = float_property(doc="radius of the ROI")
