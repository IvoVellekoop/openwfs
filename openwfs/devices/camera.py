from ..core import Detector
from harvesters.core import Harvester
import numpy as np
import astropy.units as u

"""Adapter for any GenICam/GenTL camera"""


class Camera(Detector):
    def __init__(self, **kwargs):
        """
        serial_number
        """
        self._harvester = None
        self._camera = None
        self._harvester = Harvester()
        self._harvester.add_file(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
        self._harvester.update()
        print(self._harvester.device_info_list)
        self._camera = self._harvester.create(kwargs)
        self._camera.start()
        nodes = self._camera.remote_device.node_map
        print(dir(nodes))
        data_shape = (nodes.Height.value, nodes.Width.value)
        try:
            pixel_size = [nodes.SensorPixelWidth.value, nodes.SensorPixelWidth.value] * u.um
        except AttributeError:  # the SensorPixelWidth feature is optional
            pixel_size = [1.0, 1.0] * u.um
        nodes.TriggerMode.value = 'On'
        nodes.TriggerSource.value = 'Software'
        nodes.ExposureMode.value = 'Timed'
        nodes.ExposureAuto.value = 'Off'
        #        nodes.TriggerOverlap.value = 'PreviousFrame'

        super().__init__(data_shape=data_shape, pixel_size=pixel_size, duration=nodes.ExposureTime.value * u.us)
        self.nodes = nodes  # can only set public property after initializing

    def __del__(self):
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
        if self._harvester is not None:
            self._harvester.reset()

    def _do_trigger(self):
        # while not self.nodes.TriggerReady.value:
        #     time.sleep(0.001)
        self.nodes.TriggerSoftware.execute()

    def _fetch(self, out: np.ndarray | None, *args, **kwargs) -> np.ndarray:
        buffer = self._camera.fetch()
        frame = buffer.payload.components[0].data.reshape(self.data_shape)
        if frame.size == 0:
            raise Exception('Camera returned empty frame')
        if out is not None:
            np.copyto(out, frame)
        else:
            out = frame.copy()
        buffer.queue()
        return out
