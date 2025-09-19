import warnings
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from . import safe_import

hc = safe_import("harvesters.core", "genicam")
if hc is not None:
    from harvesters.core import Harvester

from ..core import Detector


class Camera(Detector):
    """Adapter for GenICam/GenTL cameras.

    Attributes:
        _nodes: The GenICam node map of the camera.
            This map can be used to access camera properties,
            see the `GenICam/GenAPI documentation <https://www.emva.org/standards-technology/genicam/>`_
            and the `Standard Features Naming Convention
            <https://www.emva.org/wp-content/uploads/GenICam_SFNC_2_3.pdf>`_ for more details.

            The node map should not be used to set properties that are available as properties in the Camera object,
            such as ``exposure``, ``width``, ``height``, ``binning``, etc.

            Also, the node map should not be used to set properties while the camera is fetching a frame (i.e.,
            between ``trigger()`` and calling ``result()`` on the returned concurrent.futures.Future object).

    Note:
        This class is a thin wrapper around the Harvesters module,
        which is a generic adapter for GenICam/GenTL cameras.

    Example:
        >>> camera = Camera(cti_file=R"C:\\Program Files\\Basler\\pylon 7\\Runtime\\x64\\ProducerU3V.cti")
        >>> camera.exposure_time = 10 * u.ms
        >>> frame = camera.read()
    """

    def __init__(
        self,
        cti_file: str,
        serial_number: Optional[str] = None,
        multi_threaded=True,
        **kwargs,
    ):
        """
        Initialize the Camera object.

        Args:
            cti_file: The path to the GenTL producer file.
                This path depends on where the driver for the camera is installed.
                For Basler cameras, this is typically located in
                R"C:\\Program Files\\Basler\\pylon 7\\Runtime\\x64\\ProducerU3V.cti".

            serial_number: The serial number of the camera.
                When omitted, the first camera found is selected.
            **kwargs: Additional keyword arguments.
                These arguments are transferred to the node map of the camera.
        """
        self._harvester = Harvester()

        try:
            # Try to add the GenTL producer file (cti_file)
            self._harvester.add_file(cti_file, check_validity=True)
            print(f"Successfully loaded CTI file: {cti_file}")
        except Exception as e:
            # Catch any errors during the file loading process and provide a user-friendly message
            print(f"Failed to load CTI file: {cti_file}")
            print(f"Error: {str(e)}")
            print(
                "Please ensure that the CTI file exists at the specified location "
                "and that it is a valid GenTL producer file. You can download or "
                "locate the file from the camera manufacturer's website or SDK, "
                "such as the Basler pylon SDK."
            )
            raise

        self._harvester.update()

        # open the camera, use the serial_number to select the camera if it is specified.
        search_key = {"serial_number": serial_number} if serial_number is not None else None
        self._camera = self._harvester.create(search_key=search_key)
        nodes = self._camera.remote_device.node_map
        self._nodes = nodes

        # print(dir(nodes))  # for debugging, should go in a separate function

        # set triggering to 'Software', so that we can trigger the camera by calling `trigger`.
        # turn off auto exposure so that `duration` accurately reflects the required measurement time.
        settings = {
            "TriggerMode": "On",
            "TriggerSource": "Software",
            "ExposureMode": "Timed",
            "ExposureAuto": "Off",
            "BinningHorizontal": 1,
            "BinningVertical": 1,
            "OffsetX": 0,
            "OffsetY": 0,
            "Width": nodes.Width.max,
            "Height": nodes.Height.max,
        }

        # set additional properties specified in the kwargs
        settings.update(kwargs)

        for key, value in kwargs.items():
            try:
                getattr(nodes, key).value = value
            except AttributeError:
                print(f"Warning: could not set camera property {key} to {value}")

        #  Todo:
        #         automatically expose a selection of properties in the node map as
        #         properties of the Camera object.
        #

        try:
            pixel_size = [
                nodes.SensorPixelHeight.value,
                nodes.SensorPixelWidth.value,
            ] * u.um
        except AttributeError:  # the SensorPixelWidth feature is optional
            pixel_size = None

        super().__init__(
            multi_threaded=multi_threaded,
            data_shape=None,
            pixel_size=pixel_size,
            duration=None,
            latency=0.0 * u.ms,
        )
        self._camera.start()

    def __del__(self):
        if hasattr(self, "_camera"):
            self._camera.stop()
            self._camera.destroy()
        if hasattr(self, "_harvester"):
            self._harvester.reset()

    def _do_trigger(self):
        """Executes the software trigger command on the camera.

        This method is called by the trigger() method to initiate image acquisition.
        """
        self._nodes.TriggerSoftware.execute()

    def paused(self):
        """Returns a context manager for pausing the camera.

        Usage ::
            >>> with camera.paused():
            >>>     camera.nodes.SomeNode.value = 10
        """
        return _CameraPause(self._camera)

    def _fetch(self, *args, **kwargs) -> np.ndarray:
        """Fetches a frame from the camera.

        This method is called by the trigger() method (typically on a worker thread)
        to retrieve the image data after the camera has been triggered.
        Returns:
            np.ndarray: The image data from the camera.

        Raises:
            Exception: If the camera returns an empty frame.
        """
        buffer = self._camera.fetch()
        frame = buffer.payload.components[0].data.reshape(self.data_shape)
        if frame.size == 0:
            raise Exception("Camera returned an empty frame")
        data = frame.copy()
        buffer.queue()  # give back buffer to the camera driver
        return data

    @property
    def duration(self) -> Quantity[u.ms]:
        """The duration between the trigger and the end of the exposure.

        Returns ∞ · ms if hardware triggering is used."""
        # TODO: implement hardware triggering.
        return self.exposure.to(u.ms)

    @property
    def exposure(self) -> u.Quantity[u.ms]:
        """Exposure time of the camera"""
        return self._nodes.ExposureTime.value * u.us

    @exposure.setter
    def exposure(self, value: Quantity[u.ms]):
        with self.paused():
            self._nodes.ExposureTime.value = int(value.to_value(u.us))

    @property
    def binning(self) -> int:
        """Pixel binning factor.

        Note:
            setting horizontal and vertical binning separately is not supported.
        """
        return self._nodes.BinningHorizontal.value

    @binning.setter
    def binning(self, value):
        with self.paused():
            if value != self._nodes.BinningHorizontal.value:
                self._nodes.BinningHorizontal.value = int(value)
            if value != self._nodes.BinningVertical.value:
                self._nodes.BinningVertical.value = int(value)

    @property
    def top(self) -> int:
        """The vertical start position of the region of interest (in pixels).

        Note:
            The camera may round up this value to multiples of some power of 2.
        """
        return self._nodes.OffsetY.value

    @top.setter
    def top(self, value: int):
        self._set_round_up(self._nodes.OffsetX, value)

    @property
    def left(self) -> int:
        """The horizontal start position of the region of interest (in pixels).

        Note:
            The camera may round up this value to multiples of some power of 2.
        """
        return self._nodes.OffsetX.value

    @left.setter
    def left(self, value: int):
        self._set_round_up(self._nodes.OffsetX, value)

    def _set_round_up(self, node, value):
        """Sets the value of a property, rounding up to the next multiple of the increment.

        Args:
            node: The camera node to set the value for.
            value: The desired value, which will be rounded up to the next multiple of the node's increment.
        """
        inc = node.inc
        with self.paused():
            # round up value to the next multiple of inc
            node.value = int(value) + ((-value) % inc)

    @property
    def width(self) -> int:
        """Width of the camera frame, in pixels.

        Note:
            The camera may round up this value to multiples of some power of 2.
        """
        return self._nodes.Width.value

    @width.setter
    def width(self, value: int):
        self._set_round_up(self._nodes.Width, value)

    @property
    def height(self) -> int:
        """Height of the camera frame, in pixels.

        Note:
            The camera may round up this value to multiples of some power of 2.
        """
        return self._nodes.Height.value

    @height.setter
    def height(self, value: int):
        self._set_round_up(self._nodes.Height, value)

    @property
    def pixel_size(self) -> Optional[Quantity[u.um]]:
        """Physical pixel size of the camera sensor."""
        return self._pixel_size

    @property
    def data_shape(self):
        """Shape (height, width) of the data array returned by the camera."""
        return self.height, self.width

    @staticmethod
    def enumerate_cameras(cti_file: str):
        """Enumerates all cameras available through the specified GenTL producer.

        Args:
            cti_file: The path to the GenTL producer file.

        Returns:
            list: A list of device information dictionaries for all available cameras.

        Warns:
            UserWarning: If the CTI file cannot be loaded.
        """
        with Harvester() as harvester:
            try:
                harvester.add_file(cti_file, check_validity=True)
                harvester.update()
            except (OSError, FileNotFoundError):
                warnings.warn(f"Failed to load CTI file: {cti_file}")
            return harvester.device_info_list.copy()


class _CameraPause:
    """Context manager for pausing the camera."""

    def __init__(self, camera):
        self._camera = camera

    def __enter__(self):
        return self._camera.stop()

    def __exit__(self, _type, _value, _traceback):
        self._camera.start()
