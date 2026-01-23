import astropy.units as u
from astropy.units import Quantity
from openwfs.core import Actuator, Detector
import numpy as np
import time
from ..utilities.utilities import set_metadata
import shutil
import os

# ToDo
# add property to get pixel size and data_shape from microscope, currently hardcoded to the values 100 nm and (1024, 1024)
# find a way to get the actual values from the microscope software if possible

# TODO:
# all MATL functions from the evident manual

# Delete this class after testing.
# class EvidentMicroscopyImage:
#     def __init__(self, image_stack, metadata):
#         self.img = image_stack  # NumPy array (x, y, z, t, c)
#         self.metadata = metadata  # Parsed OMEXML metadata object

#     @property
#     def shape(self):
#         return self.img.shape


def is_error(result):
    """
    Check the result from Evident Microscope API call and raise an error if not OK.
    Args:
        result (str): Result string from Evident Microscope API call.
    """
    if result != "OK":
        dict_error = {
            "OK": "Success",
            "ERROR": "Failure",
            "UNAVAILABLE": "Unavailable state",
            "OUT_OF_MEMORY": "Insufficient memory",
            "INVALID_PARAMETERS": "Invalid parameters",
            "NO_FRAME": "The specified image was not found",
            "UNSUPPORTED_IMAGE": "Unsupported image",
            "UNSUPPORTED_PARAMETERS": "Unsupported parameters",
            "FILE_NOT_FOUND": "File not found",
            "UNSUPPORTED_FILE_VERSION": "Unsupported file version",
        }
        return True, RuntimeError(f"Evident Microscope returned error: {dict_error[result]}")
    else:
        return False, None


class EvidentCamera(Detector):
    """
    Camera in Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope
        saving_folder: folder to save raw OIR files, if None, raw files are not copied and stay where the microscope saves them.

    Returns:
        EvidentCamera object

    Example:
        cam = EvidentCamera(proxy)
    """

    def __init__(self, proxy, saving_folder=None):
        Detector.__init__(
            self, duration=np.inf * u.s, latency=0 * u.ms, data_shape=None, pixel_size=None
        )  # duration and pixel_size is set after triggering the camera (see _do_trigger)
        self._proxy = proxy
        self._most_recent_file_path = None  # to store the path of the most recently acquired image file
        self._saving_folder = saving_folder

    def _do_trigger(self):
        """
        Trigger the camera to acquire an image.

        Returns:
            dict: Result dictionary containing result code, wait time, and target name.
        """
        inputVal = {"executionType": "EXECUTION_TYPE_MANUAL_MAIN"}
        result = self._proxy.Protocol.startProtocol(inputVal)

        # result = self._proxy.Protocol.getAcquiringFiles()
        file_path = result["targetName"][0]["name"]
        self._most_recent_file_path = file_path  # store the most recent file path

        print("In _fetch: Acquired image saved at:", file_path)  # for debugging purposes

    def _fetch(self) -> np.ndarray:
        """
        Fetch the acquired image from the camera and convert to numpy array.
        """
        file_path = self._most_recent_file_path

        ######## temporary solution to access network path ########
        # Normalize slashes so replacement works
        # normalized_path = file_path.replace("/", "\\")  # convert forward to backslashes
        # network_path = normalized_path.replace("D:\\Confocal Images\\Arnon (BMPI)", r"\\CZC4497M36\Arnon (BMPI)")
        # file_path = network_path
        # print("Accessing image from network path:", file_path)  # for debugging purposes
        #####################################################################

        while self.busy():
            print("waiting for imaging to complete...")
            time.sleep(0.2)

        if self._saving_folder is not None:
            # copy raw OIR files to predefined saving folder
            if not os.path.isdir(self._saving_folder):
                raise FileNotFoundError(f"Saving folder does not exist: {self._saving_folder}")

            destination = os.path.join(self._saving_folder, os.path.basename(file_path))
            shutil.copy2(file_path, destination)

        image_stack = self.read_evident_image(file_path)

        # self.wait() # waiting for imaging to complete - \this function is not working, it waits for fetch to finish according to the base class implementation, and since fetch is called within read, it causes a loop of waiting.

        # time.sleep(2)  # wait for file to be fully written, Find better solution later

        return image_stack

    @staticmethod
    def get_image_size(proxy, image_id):
        inputVal = {"imageId": image_id}
        result = proxy.IDA.getImageSize(inputVal)
        err = is_error(result["result"])
        if err[0]:
            return err
        width = result["width"]
        height = result["height"]
        return False, width, height

    @staticmethod
    def get_image_data(proxy, image_id):
        sizes = get_image_size(proxy, image_id)
        if sizes[0]:
            return sizes
        width = sizes[1]
        height = sizes[2]
        rect = {"x": 0, "y": 0, "width": width, "height": height}
        inputVal = {"imageId": image_id, "rect": rect, "compressType": "NONE"}
        result = proxy.IDA.getImageBody(inputVal)
        err = is_error(result["result"])
        if err[0]:
            return err
        image = result["data"].data
        while result["continue"]:
            inputVal = {"imageId": image_id}
            result = proxy.IDA.getNextImageBody(inputVal)
            err = is_error(result["result"])
            if err[0]:
                return err
            image = image + result["data"].data
        return False, np.frombuffer(image, dtype=np.uint16).reshape((height, width))

    @staticmethod
    def get_axes_on_image(proxy, areaId, labels_axes):
        assert labels_axes.shape == (3,)
        inputVal = {"areadId": areaId, "key": "Axes"}
        result = proxy.IDA.getProperty(inputVal)
        err = is_error(result["result"])
        if err[0]:
            return err
        out = np.array([0, 0, 0], dtype=bool)
        axes = result["value"]
        for i in np.arange(3):
            if labels_axes[i] in axes:
                out[i] = True

        return False, out

    @staticmethod
    def get_image_id(proxy, area_id, bool_axes, labels_axes, index_axes, channel_list):
        axis_info = []
        assert bool_axes.shape == index_axes.shape
        for exists, index, label in zip(bool_axes, index_axes, labels_axes):
            if exists:
                axis_info.append({"axisName": label, "index": index})
        inputVal = {"areaId": area_id, "channelId": channel_list[0], "axisInfo": axis_info}
        result = proxy.IDA.getImage(inputVal)
        err = is_error(result["result"])
        if err[0]:
            return err
        return False, result["imageId"]

    @staticmethod
    def get_area_id(proxy, group_id):
        input_val = {"groupId": group_id, "level": 0, "area": 0}
        result = proxy.IDA.getArea(input_val)
        err = is_error(result["result"])
        if err[0]:
            return err
        return False, result["areaId"]

    @staticmethod
    def get_channel_list(proxy, area_id):
        input_val = {"areaId": area_id,
                     "key": "ChannelIdList"}
        result = proxy.IDA.getChannelList(input_val)
        err = is_error(result["result"])
        if err[0]:
            return err
        return False, result["value"]


    @staticmethod
    def get_image(proxy, filepath):
        # TODO: need to update to handle stack of channels, time, and z
        inputVal = {"filename": filepath}
        result = proxy.IDA.open(inputVal)
        error = None
        error_file = is_error(result["result"])
        if not error_file[0]:
            fileId = result["fileId"]
            group_id = proxy.IDA.getGroup({"fileId": fileId, "group": 0})
            group_id_err = is_error(group_id["result"])
            if not group_id_err[0]:
                area_id = EvidentCamera.get_area_id(proxy, result["groupId"])
                if not area_id[0]:
                    channel_list = EvidentCamera.get_channel_list(proxy, area_id[1])
                    if not channel_list[0]:
                        image_id = EvidentCamera.get_image_id(
                            proxy,
                            area_id[1],
                            np.array([False, False, False]),
                            np.array(["LAMBDA", "ZSTACK", "TIMELAPSE"]),
                            np.array([0, 0, 0]),
                            channel_list[1],
                        )
                        if not image_id[0]:
                            data = EvidentCamera.get_image_data(proxy, image_id)
                            proxy.IDA.releaseImage({"imageId": image_id[1]})
                        else:
                            error = image_id[1]
                    else:
                        error = channel_list[1]
                    proxy.IDA.releaseArea({"areaId": area_id[1]})
                else:
                    error = area_id[1]
                proxy.IDA.releaseGroup({"groupId": result["groupId"]})
            else:
                error = group_id_err[1]
            proxy.IDA.close({"fileId": fileId})
        else:
            error = error_file[1]
        if error is not None:
            raise RuntimeError("Failed to get image. Error from evident microscope: {}".format(error))


    @staticmethod
    def read_evident_image(file_path) -> np.ndarray:
        """
        Static method to read an Evident image file and return the image data and metadata.
        The metadata is stored in the dtype and can be found using the get_metadata function.

        Example usage:
        from private_openwfs.devices.stages_evident import EvidentCamera as EvidentPrivate
        img = EvidentPrivate.read_evident_image(file_path)
        from private_openwfs.utilities.utilities import get_metadata
        meta_data = get_metadata(img)

        Args:
            file_path (str): Path to the Evident image file.
        Returns:
            image (np.ndarray): Acquired image data.
        """

        metadata = bioformats.get_omexml_metadata(file_path)
        meta_data = bioformats.OMEXML(
            metadata
        )  # takes that raw XML string and parses it into a Python object that’s easier to work with. The OMEXML class understands the OME-XML schema and gives you structured access.

        # Extract dimensions
        size_x = meta_data.image().Pixels.SizeX
        size_y = meta_data.image().Pixels.SizeY
        size_z = meta_data.image().Pixels.SizeZ
        size_c = meta_data.image().Pixels.SizeC
        size_t = meta_data.image().Pixels.SizeT

        # Initialize array in (x, y, z, time, channel) order
        image_stack = np.zeros((size_x, size_y, size_z, size_t, size_c), dtype=np.uint16)

        with bioformats.ImageReader(file_path) as reader:
            # Read and fill the array
            for t in range(size_t):
                for c in range(size_c):
                    for z in range(size_z):
                        img = reader.read(c=c, z=z, t=t, rescale=False)
                        image_stack[:, :, z, t, c] = img

        image_stack = set_metadata(image_stack, meta_data)
        return image_stack

    def busy(self) -> bool:
        """
        Returns true if the device is measuring else false.
        If get protocol progress returns a state other than IDLING, the device is busy.
        """
        # print(" busy function is used in EvidentCamera to check if the camera is measuring.")
        return self.get_imaging_progress()["state"] != "IDLING"

    # we are using the busy function to check if the camera is measuring, so duration property is not needed. it is not possible to query the expected duration from the microscope
    # @property
    # def duration(self) -> Quantity[u.s]:
    #     """
    #     Get the duration of the current imaging protocol.
    #     The duration is set after triggering the camera (see _do_trigger).
    #     """
    #     return self._duration

    @property  # this works as long as size of the image is set to 1024 x 1024 and 2 channels in the microscope software
    def data_shape(self) -> tuple[int, int]:
        """
        Get the shape of the acquired image data (height, width).
        """
        height, width, z, t, c = 1024, 1024, 1, 1, 2  # default values
        return (height, width, z, t, c)

    @property
    def pixel_size(
        self,
    ) -> Quantity[u.nm]:  # a random number for now, need to find a way to get the actual pixel size from the microscope
        """
        Get the pixel size of the acquired image data.
        """
        pixel_size_nm = 100  # example value in nm, replace with actual value if available
        return pixel_size_nm * u.nm

    def get_imaging_progress(self, execution_type: str = "EXECUTION_TYPE_MANUAL_MAIN") -> dict:
        """
        Get the progress of the current imaging protocol.

        Args:
            execution_type (str): Type of execution. Options are "EXECUTION_TYPE_MANUAL_MAIN", "EXECUTION_TYPE_SM", or "EXECUTION_TYPE_MATL".
                                  Default is "EXECUTION_TYPE_MANUAL_MAIN".

        Returns:
            dict: Result dictionary containing imaging progress information.
        """
        inputVal = {"executionType": execution_type}
        return self._proxy.Protocol.getProtocolProgress(inputVal)

    def stop_imaging(self, execution_type: str = "EXECUTION_TYPE_MANUAL_MAIN") -> dict:
        """
        Stop imaging on the microscope.

        Args:
            execution_type (str): Type of execution. Options are "EXECUTION_TYPE_MANUAL_MAIN" or "EXECUTION_TYPE_MATL".
                                  Default is "EXECUTION_TYPE_MANUAL_MAIN".

        Returns:
            dict: Result dictionary containing result code, wait time, and target name.
        """
        inputVal = {"executionType": execution_type}
        return self._proxy.Protocol.stopProtocol(inputVal)


class EvidentXYStage(Actuator):
    """
    Stage in Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope

    Returns:
        EvidentXYStage object with .xy property to get and set the stage position.

    Example:
        stage = EvidentXYStage(proxy)
        stage.xy = (10 *u.nm , 20*u.nm) # move stage to (10, 20) in nm
        x, y = stage.xy # get current stage position
    """

    def __init__(self, proxy):
        Actuator.__init__(
            self, duration=np.inf * u.ms, latency=0 * u.ms
        )  # duration is infinite since we don't know when the stage will reach the target position, busy() method will handle that.
        self._proxy = proxy
        self._target_xy = None  # target position to move to (used in busy() method), in nm as int
        self._step = 100 * u.nm  # rounding resolution

    @staticmethod
    def round_quantity(q, step):
        # convert to the same unit as step
        q_in_step = q.to(step.unit)
        # round value
        rounded_value = np.round(q_in_step.value / step.value) * step.value
        # return Quantity with correct unit
        return rounded_value * step.unit

    def busy(self) -> bool:
        """
        Returns true if the device is moving else false.
        If stage position is not equal to target position, the device is busy.
        """
        print(" busy function is used in EvidentXYStage to check if the stage is moving.")
        print("Current stage position (x, y):", self.xy)
        print("Target stage position (x, y):", self._target_xy)
        time.sleep(0.5)  # to avoid spamming the proxy with requests
        return tuple(int(q.value) for q in self.xy) != self._target_xy

    @property
    def xy(self) -> tuple[Quantity[u.nm], Quantity[u.nm]]:
        # get current xy stage position
        inpuVal = {"settingId": "XY_STAGE_POSITION_SETTING"}
        result = self._proxy.Parameter.getParameter(inpuVal)
        stage_x = result["x"]
        stage_y = result["y"]
        return stage_x * u.nm, stage_y * u.nm

    @xy.setter
    def xy(self, value: tuple[Quantity, Quantity]):
        super()._start()  # wait for all previous commands to finish
        x, y = value
        # print(x)
        # print(self._step)
        x_rounded = self.round_quantity(x, self._step)
        y_rounded = self.round_quantity(y, self._step)
        self._target_xy = (int(x_rounded.to(u.nm).value), int(y_rounded.to(u.nm).value))
        x = int(x_rounded.to(u.nm).value)
        y = int(y_rounded.to(u.nm).value)
        inputVal = {"settingId": "XY_STAGE_POSITION_SETTING", "x": x, "y": y, "escapeEnabled": False}
        self._proxy.Parameter.setParameter(inputVal)

    @property
    def x(self) -> Quantity[u.nm]:
        # get current x stage position
        return self.xy[0]

    @x.setter
    def x(self, x: Quantity):
        self.xy = x, self.y

    @property
    def y(self) -> Quantity[u.nm]:
        # get current y stage position
        return self.xy[1]

    @y.setter
    def y(self, y: Quantity):
        self.xy = self.x, y


class EvidentMicroscope:  # (Processor): How to calculate duration and latency, since it depend on both camera and stage?
    """
    Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope
        saving_folder: folder to save raw OIR files, if None, raw files are not copied and stay where the microscope saves them.

    Returns:
        EvidentMicroscope object with .camera and .stage properties.

    Example:
        mic = EvidentMicroscope(proxy)
        cam = mic.camera
        stage = mic.stage
    """

    def __init__(self, proxy, saving_folder=None):
        self._proxy = proxy
        self.cam = EvidentCamera(proxy, saving_folder)
        self.stage_xy = EvidentXYStage(proxy)
        # super().__init__(self.cam, self.stage_xy)

    @property
    def version(self) -> str:
        """
        Get the product version.

        Returns:
            str: Version string, e.g., "1.1.1.1"
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_VERSION_SETTING"}
        return self._proxy.Parameter.getParameter(inputVal)["version"]

    @property
    def t_stack_enabled(self) -> bool:
        """
        Get Enable/Disable status of T stack

        Returns:
            bool: True if T stack is enabled, False if disabled.
        """
        inputVal = {"settingId": "LSM_TIME_LAPS_ENABLE_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["enable"]

    @t_stack_enabled.setter
    def t_stack_enabled(self, enable: bool):
        """
        Set Enable/Disable of T stack

        Args:
            enable (bool): True to enable T stack, False to disable.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_TIME_LAPS_ENABLE_SETTING", "enable": enable}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set T stack enable status. Result code: {}".format(result["result"]))

    def load_scan_parameters(self, file_path: str):
        """
        Load scan parameters from the specified file path.

        Args:
            file_path (str): File path to the target image for loading.

        Returns:
            dict: file ID.

        """
        inputVal = {"targetName": file_path}
        result = self._proxy.Parameter.loadParameter(inputVal)
        if result["result"] != "OK":
            raise RuntimeError("Failed to load image. Result code: {}".format(result["result"]))

    def get_channel_list(self) -> list[str]:
        """
        Get the list of available channels.

        Returns:
            list[str]: List of channel Ids.
        """
        inputVal = {"settingId": "LSM_CHANNEL_LIST_SETTING"}
        return self._proxy.Parameter.getParameter(inputVal)["channelList"]

    def get_channel_enabled(self, channel_id: str) -> bool:
        """
        Get Enable/Disable of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            bool: True if channel is enabled, False if disabled.
        """
        inputVal = {"settingId": "LSM_CHANNEL_ENABLE_SETTING", "channelId": channel_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["enable"]

    def set_channel_enabled(self, channel_id: str, enable: bool):
        """
        Set Enable/Disable of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.
            enable (bool): True to enable channel, False to disable.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_CHANNEL_ENABLE_SETTING", "channelId": channel_id, "enable": enable}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set channel enable status. Result code: {}".format(result["result"]))

    def get_channel_hv(self, channel_id: str) -> int:
        """
        Get HV of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            int: HV value of the specified channel.
        """
        inputVal = {"settingId": "LSM_CHANNEL_HV_SETTING", "channelId": channel_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["hv"]

    def set_channel_hv(self, channel_id: str, hv: int):
        """
        Set HV of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.
            hv (int): HV value to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_CHANNEL_HV_SETTING", "channelId": channel_id, "hv": hv}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set channel HV. Result code: {}".format(result["result"]))

    def get_channel_gain(self, channel_id: str) -> float:
        """
        Get Gain of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            float: Gain value of the specified channel.
        """
        inputVal = {"settingId": "LSM_CHANNEL_GAIN_SETTING", "channelId": channel_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        print(result)
        return result["gain"]

    def set_channel_gain(self, channel_id: str, gain: float):
        """
        Set Gain of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.
            gain (float): Gain value to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_CHANNEL_GAIN_SETTING", "channelId": channel_id, "gain": gain}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set channel Gain. Result code: {}".format(result["result"]))

    def get_channel_offset(self, channel_id: str) -> int:
        """
        Get Offset of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            int: Offset value of the specified channel.
        """
        inputVal = {"settingId": "LSM_CHANNEL_OFFSET_SETTING", "channelId": channel_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["offset"]

    def set_channel_offset(self, channel_id: str, offset: int):
        """
        Set Offset of Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.
            offset (int): Offset value to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_CHANNEL_OFFSET_SETTING", "channelId": channel_id, "offset": offset}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set channel Offset. Result code: {}".format(result["result"]))

    def get_laser_list_settings(self) -> list[str]:
        """
        Get the list of available lasers.

        Returns:
            list[str]: List of laser Ids.
            list is a struct with fields:
                imagingDeviceMode: Specify IMAGING.
                laserScannerMode: Specify MAIN.
                phaseId: Phase ID
                laserType: Laser type SPE or MPE
                laserWaveLength: Laser wavelength
                laserId: Laser Id
        """
        inputVal = {"settingId": "LSM_LASER_LIST_SETTING"}
        return self._proxy.Parameter.getParameter(inputVal)["laserList"]

    def get_channel_laser(self, channel_id: str) -> list[str]:
        """
        Get the Laser set to Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            list[str]: List of laser settings for the specified channel.
        """
        inputVal = {"settingId": "LSM_CHANNEL_LASER_SETTING", "channelId": channel_id}
        return self._proxy.Parameter.getParameter(inputVal)["laser"]["laserId"]

    def set_channel_laser(self, channel_id: str, laser_settings: list[dict]):
        """
        Set the Laser to Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.
            laser_settings (list[dict]): List of laser settings to set for the specified channel.
                Each dict should have keys:
                    imagingDeviceMode (str): Specify IMAGING.
                    laserScannerMode (str): Specify MAIN.
                    laserId (str): Specify the laserId acquired by LSM_LASER_LIST_SETTING.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_CHANNEL_LASER_SETTING", "channelId": channel_id, "laser": laser_settings}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set channel Laser. Result code: {}".format(result["result"]))

    def get_laser_enabled(self, laser_id: str) -> bool:
        """
        Get Enable/Disable of Laser

        Args:
            laser_id (str): Laser Id acquired by LSM_LASER_LIST_SETTING.

        Returns:
            bool: True if laser is enabled, False if disabled.
        """
        inputVal = {"settingId": "LSM_LASER_ENABLE_SETTING", "laserId": laser_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        print(result)
        return result["enable"]

    def set_laser_enabled(self, laser_id: str, enable: bool):
        """
        Set Enable/Disable of Laser

        Args:
            laser_id (str): Laser Id acquired by LSM_LASER_LIST_SETTING.
            enable (bool): True to enable laser, False to disable.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_LASER_ENABLE_SETTING", "laserId": laser_id, "enable": enable}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set laser enable status. Result code: {}".format(result["result"]))

    def get_laser_intensity(self, laser_id: str) -> float:
        """
        Get the Laser intensity

        Args:
            laser_id (str): Laser Id acquired by LSM_LASER_LIST_SETTING.

        Returns:
            float: Laser intensity of the specified laser.
        """
        inputVal = {"settingId": "LSM_LASER_INTENSITY_SETTING", "laserId": laser_id}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["laserIntensity"]

    def set_laser_intensity(self, laser_id: str, intensity: float):
        """
        Set the Laser intensity

        Args:
            laser_id (str): Laser Id acquired by LSM_LASER_LIST_SETTING.
            intensity (float): Laser intensity to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_LASER_INTENSITY_SETTING", "laserId": laser_id, "laserIntensity": intensity}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set laser intensity. Result code: {}".format(result["result"]))

    @property
    def objective(self) -> str:
        """
        Get the current objective lens name.

        Returns:
            str: Name of the current objective lens.
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_CURRENT_SETTING"}
        # print("The current objective lens is:", result["revolver"]["lensName"], "with magnification of:", result["revolver"]["magnification"])
        return self._proxy.Parameter.getParameter(inputVal)["revolver"]

    @objective.setter
    def objective(self, revolver_index: int):
        """
        Set the objective lens by specifying the revolver index.

        Args:
            revolver_index (int): Index of the nosepiece to be changed.
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_CURRENT_SETTING", "revolverIndex": revolver_index}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":  # check if the result is OK, meaning the objective was set successfully
            raise RuntimeError("Failed to set objective lens. Result code: {}".format(result["result"]))

    @property
    def objective_list(self) -> list[str]:
        """
        Get the list of available objective lenses.

        Returns:
            list[str]: List of objective lens names.
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_LIST_SETTING"}
        return self._proxy.Parameter.getParameter(inputVal)["revolvers"]

    @property
    def z(self) -> Quantity[u.nm]:
        """
        Get current z stage position
        """
        inputVal = {"settingId": "Z_STAGE_POSITION_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        stage_z = result["stagePosition"]
        return stage_z * u.nm

    @z.setter
    def z(self, z: Quantity):
        """
        Set z stage position
        """
        z = int(z.to(u.nm).value)
        inputVal = {"settingId": "Z_STAGE_POSITION_SETTING", "stagePosition": z}
        self._proxy.Parameter.setParameter(inputVal)

    @property
    def z_stack_enabled(self) -> bool:
        """
        Get Enable/Disable status of Z stack

        Returns:
            bool: True if Z stack is enabled, False if disabled.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_ENABLE_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["enable"]

    @z_stack_enabled.setter
    def z_stack_enabled(self, enable: bool):
        """
        Set Enable/Disable of Z stack

        Args:
            enable (bool): True to enable Z stack, False to disable.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_ENABLE_SETTING", "enable": enable}
        result = self._proxy.Parameter.setParameter(inputVal)
        if result["result"] != "OK":
            raise RuntimeError("Failed to set Z stack enable status. Result code: {}".format(result["result"]))

    @property
    def z_stack_slice_num(self) -> int:
        """
        Get the number of slices of Z stack.

        Returns:
            int: Number of slices.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_SLICE_NUM_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["sliceNum"]

    @z_stack_slice_num.setter
    def z_stack_slice_num(self, slice_num: int):
        """
        Set the number of slices of Z stack.

        Args:
            slice_num (int): Number of slices to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_SLICE_NUM_SETTING", "sliceNum": slice_num}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set Z stack slice number. Result code: {}".format(result["result"]))

    @property
    def z_stack_step_size(self) -> int:
        """
        Get the step size of Z stack.

        Returns:
            int: Step size in nm.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_STEP_SIZE_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        return result["stepSize"] * u.nm

    @z_stack_step_size.setter
    def z_stack_step_size(self, step_size: int):
        """
        Set the step size of Z stack.

        Args:
            step_size (Quantity): Step size in nm to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_STEP_SIZE_SETTING", "stepSize": step_size.to(u.nm).value}
        result = self._proxy.Parameter.setParameter(inputVal)

        if result["result"] != "OK":
            raise RuntimeError("Failed to set Z stack step size. Result code: {}".format(result["result"]))
