import numpy as np
from base_device_properties import *


class Feedback:
    def __init__(self, source, slm):
        self.measurements_pending = 0
        self._measurements = None
        self._measurements_flat = None
        self.shape = None
        self.source = source
        self.slm = slm
        self.M = None
        self.n = None

    def __del__(self):
        while self.measurements_pending > 0:
            self.process_data()

    def reserve(self, shape):
        self.n = 0
        self.M = np.prod(self.source.data_shape)
        self._measurements = np.zeros((*shape, self.M), dtype="float32")
        self._measurements_flat = np.reshape(self._measurements, (np.prod(shape), self.M))

    def measure(self):
        # Update the SLM. If the SLM was reserved, first wait until the measurement_time - idle_time has passed before
        # flipping the buffers. Don't wait for the SLM to stabilize
        self.slm.update(wait=False)

        # we can use this time to process a previous measurement (if any)
        self.process_data()

        self.slm.wait()  # wait for the image on the SLM to stabilize
        self.source.trigger()  # trigger the camera
        self.measurements_pending += 1

        measurement_time = self.source.measurement_time
        if measurement_time is not None:
            # default fast wavefront shaping, continue processing (and even start next frame) during measurement
            self.slm.reserve(measurement_time)
        else:
            # measurement time not known, wait till end of measurement
            self.process_data()

    def process_data(self):
        if self.measurements_pending > 0:
            data = np.array(self.source.read(), dtype="float32", copy=False)
            self.measurements_pending -= 1
            self._measurements_flat[self.n, :] = data.flat

    @property
    def measurements(self):
        while self.measurements_pending > 0:
            self.process_data()
        return self._measurements


class SimpleCameraFeedback(Feedback):
    def __init__(self, camera, slm, roi_x, roi_y, roi_radius):
        roi_signal = SingleRoi(camera, x=roi_x, y=roi_y, radius=roi_radius)
        super().__init__(roi_signal, slm)


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
