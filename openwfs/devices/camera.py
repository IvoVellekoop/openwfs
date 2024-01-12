from ..core import Detector
from harvesters.core import Harvester
import numpy as np
import astropy.units as u
from contextlib import contextmanager

"""Adapter for GenICam/GenTL cameras"""


class Camera(Detector):
    """
        Adapter for GenICam/GenTL cameras.

        Attributes:
            nodes: The GenICam node map of the camera.
                This map can be used to access camera properties,
                see the `GenICam/GenAPI documentation <https://www.emva.org/standards-technology/genicam/>`_
                and the `Standard Features
Naming Convention <https://www.emva.org/wp-content/uploads/GenICam_SFNC_2_3.pdf>` for more details.

                The node map should not be used to set properties that are available as properties in the Camera object,
                such as `duration` (exposure time), `width`, `height`, `binning`, etc.

                Also, the node map should not be used to set properties while the camera is fetching a frame (i.e.,
                between `trigger()` and calling `result()` on the returned concurrent.futures.Future obect.

        Note:
            This class is a thin wrapper around the Harvesters module,
            which is a generic adapter for GenICam/GenTL cameras.

        Example:
            camera = Camera(serial_number='12345678')
            camera.start()
            frame = camera.capture_frame()
            camera.stop()'
        """

    def __init__(self, cti_file: str, serial_number: str | None = None, **kwargs):
        """
            Initialize the Camera object.

            Args:
                cti_file: The path to the GenTL producer file.
                    This path depends on where the driver for the camera is installed.
                    For Basler cameras, this is typically located in
                    R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti".

                serial_number: The serial number of the camera.
                    When omitted, the first camera found is selected.
                **kwargs: Additional keyword arguments.
                    These arguments are transferred to the node map of the camera.

            Todo:
                automatically expose a selection of properties in the node map as
                properties of the Camera object.

            Note:
                The `serial_number` argument is required to specify the serial number of the camera.
        """
        self._harvester = Harvester()
        self._harvester.add_file(cti_file, check_validity=True)
        self._harvester.update()

        print(self._harvester.device_info_list)  # for debugging, should go in a separate function

        # open the camera, use the serial_number to select the camera if it is specified.
        search_key = {'serial_number': serial_number} if serial_number is not None else None
        self._camera = self._harvester.create(search_key=search_key)
        nodes = self._camera.remote_device.node_map

        print(dir(nodes))  # for debugging, should go in a separate function

        # set triggering to 'Software', so that we can trigger the camera by calling `trigger`.
        # turn off auto exposure so that `duration` accurately reflects the required measurement time.
        nodes.TriggerMode.value = 'On'
        nodes.TriggerSource.value = 'Software'
        nodes.ExposureMode.value = 'Timed'
        nodes.ExposureAuto.value = 'Off'
        nodes.BinningHorizontal.value = 1
        nodes.BinningVertical.value = 1
        nodes.OffsetX.value = 0
        nodes.OffsetY.value = 0
        nodes.Width.value = nodes.Width.max
        nodes.Height.value = nodes.Height.max

        # set additional properties specified in the kwargs
        for key, value in kwargs.items():
            try:
                setattr(nodes, key, value)
            except AttributeError:
                print(f'Warning: could not set camera property {key} to {value}')

        data_shape = (nodes.Height.value, nodes.Width.value)
        try:
            pixel_size = [nodes.SensorPixelWidth.value, nodes.SensorPixelWidth.value] * u.um
        except AttributeError:  # the SensorPixelWidth feature is optional
            pixel_size = [1.0, 1.0] * u.um

        super().__init__(data_shape=data_shape, pixel_size=pixel_size, duration=nodes.ExposureTime.value * u.us)
        self.nodes = nodes  # can only set public property after initializing
        self._camera.start()

    def __del__(self):
        if hasattr(self, '_camera'):
            self._camera.stop()
            self._camera.destroy()
        if hasattr(self, '_harvester'):
            self._harvester.reset()

    def _do_trigger(self):
        self.nodes.TriggerSoftware.execute()

    def paused(self):
        """Returns a context manager for pausing the camera.
        Usage:
            with camera.paused():
                camera.nodes.SomeNode.value = 10
        """
        return _CameraPause(self._camera)

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

    def _update_roi(self):
        self._data_shape = (self.nodes.Height.value, self.nodes.Width.value)

    @property
    def binning(self) -> int:
        """
        Pixel binning factor
        Note:
            setting horizontal and vertical binning separately is not supported.
        """
        return self.nodes.BinningHorizontal.value

    @binning.setter
    def binning(self, value):
        with self.paused():
            if value != self.nodes.BinningHorizontal.value:
                self.nodes.BinningHorizontal.value = int(value)
            if value != self.nodes.BinningVertical.value:
                self.nodes.BinningVertical.value = int(value)
        self._update_roi()

    @property
    def top(self) -> int:
        """
        Top position of the camera frame.

        Returns:
        int: The top position of the camera frame.
        """
        return self.nodes.OffsetY.value

    @top.setter
    def top(self, value: int):
        self._set_round_up(self.nodes.OffsetX, value)
        self._update_roi()

    @property
    def left(self) -> int:
        """
        Left position of the camera frame.

        Returns:
        int: The left position of the camera frame.
        """
        return self.nodes.OffsetX.value

    @left.setter
    def left(self, value: int):
        self._set_round_up(self.nodes.OffsetX, value)
        self._update_roi()

    def _set_round_up(self, node, value):
        """Sets the value of a property, rounding up to the next multiple of the increment."""
        inc = node.inc
        with self.paused():
            # round up value to the next multiple of inc
            node.value = int(value) + ((-value) % inc)

    @property
    def width(self) -> int:
        """
        Width of the camera frame.

        Returns:
        int: The width of the camera frame.
        """
        return self.nodes.Width.value

    @width.setter
    def width(self, value: int):
        self._set_round_up(self.nodes.Width, value)
        self._update_roi()

    @property
    def height(self) -> int:
        """
        Height of the camera frame.

        Returns:
        int: The height of the camera frame.
        """
        return self.nodes.Height.value

    @height.setter
    def height(self, value: int):
        self._set_round_up(self.nodes.Height, value)
        self._update_roi()


class _CameraPause:
    def __init__(self, camera):
        self._camera = camera

    def __enter__(self):
        return self._camera.stop()

    def __exit__(self, _type, _value, _traceback):
        self._camera.start()
