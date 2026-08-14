import astropy.units as u
from astropy.units import Quantity
from ..core import Actuator, Detector
from ..utilities.utilities import round_quantity
import numpy as np
import time

import os
import xmlrpc.client
import itertools
import time

# TO DO:
# Check if it is possible to get the image data while imaging. E.g. for multiple time stacks get the data while imaging the next frame.
# Lasers measured on different channels, not clear that the ND array structure is correct for that case
# add all the MATL imaging functions


class EvidentError:
    """
    Simple class to handle the errors from the Evident SDK calls.
    """

    def __init__(self, is_error: bool, error):
        self.is_error = is_error
        self.error = error

    """
        If there is an error, raise the error. Otherwise, do nothing.
    """

    def throw_if_error(self):
        if self.is_error:
            raise self.error


def get_evident_error(output_command):
    """
    Check the result from Evident Microscope API call and returns the error if not OK. The error is not throw by this function, it is returned to be handled by the caller.
    Args:
        err (EvidentError): Class with error message and boolean indicating if there is an error or not.
    """

    if output_command["result"] != "OK":
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
        return EvidentError(
            True,
            RuntimeError(
                f"Evident Microscope returned error: {dict_error[output_command['result']]}. The output of the command was : {output_command}"
            ),
        )
    else:
        return EvidentError(False, None)


class EvidentCamera(Detector):
    """
    Camera in Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope

    Returns:
        EvidentCamera object

    Example:
        cam = EvidentCamera(proxy)
    """

    def __init__(self, proxy):
        Detector.__init__(
            self, duration=np.inf * u.s, latency=0 * u.ms, data_shape=None, pixel_size=None
        )  # duration and pixel_size is set after triggering the camera (see _do_trigger)

        self._proxy = proxy
        self._most_recent_file_path = None  # to store the path of the most recently acquired image file
        self.images_saved = 0

    def _do_trigger(self):
        """
        Trigger the camera to acquire an image.

        Returns:
            dict: Result dictionary containing result code, wait time, and target name.
        """
        inputVal = {"executionType": "EXECUTION_TYPE_MANUAL_MAIN"}
        result = self._proxy.Protocol.startProtocol(inputVal)
        error = get_evident_error(result)
        error.throw_if_error()

        # result = self._proxy.Protocol.getAcquiringFiles()
        file_path = result["targetName"][0]["name"]
        self._most_recent_file_path = file_path  # store the most recent file path
        self._planned_imaging_time = result["wait"] * u.s

    def _fetch(self) -> np.ndarray:
        """
        Blocks the thread until the camera has finished acquiring the image and retrieves the image data returned as a numpy array.
        """
        while self.busy():
            time.sleep(0.2)

        image, pixel_sizes = self.get_image(self._proxy, self._most_recent_file_path)

        self._pixel_size = pixel_sizes[0:2] * u.um  # This needs to be changes to hanbdle multidimensional images

        return image

    @staticmethod
    def get_image_size(proxy, image_id):
        """
        Get the size of the image with the given image ID using the Evident Microscope API.

        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            image_id: ID of the image to get the size of
        Returns:
            tuple: (EvidentError, width, height)
        """
        inputVal = {"imageId": image_id}
        result = proxy.IDA.getImageSize(inputVal)
        err = get_evident_error(result)
        if err.is_error:
            return err, 0, 0
        width = result["width"]
        height = result["height"]
        return err, width, height

    @staticmethod
    def get_image_data(proxy, image_id):
        """
        Get the image data with the given image ID using the Evident Microscope API.

        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            image_id: ID of the image to get the data of

        Returns:
            tuple: (EvidentError, image data as numpy array)
        """
        err, width, height = EvidentCamera.get_image_size(proxy, image_id)
        if err.is_error:
            return err, None

        rect = {"x": 0, "y": 0, "width": width, "height": height}
        inputVal = {"imageId": image_id, "rect": rect, "compressType": "NONE"}
        result = proxy.IDA.getImageBody(inputVal)
        err = get_evident_error(result)

        if err.is_error:
            return err, None

        image = result["data"].data
        i = 0
        while result["continue"]:
            i = i + 1
            inputVal = {"imageId": image_id}
            result = proxy.IDA.getNextImageBody(inputVal)
            err = get_evident_error(result)
            if err.is_error:
                return err, None

            image = image + result["data"].data

        return err, np.frombuffer(image, dtype=np.uint16).reshape(height, width)

    @staticmethod
    def get_axes_on_image(proxy, areaId, labels_axes):
        """
        Get the axes present on the image with the given area ID using the Evident Microscope API.
        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            areaId: ID of the area to get the axes of
            labels_axes: np.ndarray of str
                Array with the labels of the axes to check for presence on the image. E.g. np.array(["LAMBDA", "ZSTACK", "TIMELAPSE"])
        Returns:
            tuple: (EvidentError, axes_bool, size_bool)
                axes_bool: np.ndarray of bool
                    Array indicating the presence of each axis in labels_axes on the image.
                size_bool: np.ndarray of int
                    Array with the size of each axis in labels_axes on the image. If the axis is not present, the size is set to 1.
        """
        assert labels_axes.shape == (3,)
        inputVal = {"areaId": areaId, "key": "Axes"}
        result = proxy.IDA.getProperty(inputVal)
        err = get_evident_error(result)
        if err.is_error:
            return err, None, None
        axes_bool = np.array([0, 0, 0], dtype=bool)
        size_bool = np.array([1, 1, 1], dtype=int)
        axes = result["value"]
        for i in np.arange(3):
            axis_name = labels_axes[i]
            if axis_name in axes:
                axes_bool[i] = True
                inputVal = {"areaId": areaId, "key": "AxisMaxSize", "additional": {"axisName": str(axis_name)}}
                result_size = proxy.IDA.getProperty(inputVal)
                err_size = get_evident_error(result_size)
                if err_size.is_error:
                    return err_size, None, None

                size_bool[i] = int(result_size["value"][0])
        return err, axes_bool, size_bool

    @staticmethod
    def get_image_id(proxy, area_id, labels_axes, index_axes, channel):
        """
        Get the image ID with the given area ID, axes labels, axes indices, and channel using the Evident Microscope API.
        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            area_id: ID of the area to get the image ID of
            labels_axes: np.ndarray of str
                Array with the labels of the axes to get the image ID for. E.g. np.array(["LAMBDA", "ZSTACK", "TIMELAPSE"])
            index_axes: np.ndarray of int
                Array with the indices of the axes to get the image ID for. E.g. np.array([0, 1, 2])
            channel: str
                Channel ID to get the image ID for.
        Returns:
            tuple: (EvidentError, image ID)
        """

        assert labels_axes.shape == index_axes.shape

        axis_info = []
        for index, label in zip(index_axes, labels_axes):
            axis_info.append({"axisName": str(label), "index": int(index)})

        inputVal = {"areaId": area_id, "channelId": channel, "axisInfo": axis_info}

        result = proxy.IDA.getImage(inputVal)
        err = get_evident_error(result)
        if err.is_error:
            return err, None
        return err, result["imageId"]

    @staticmethod
    def get_area_id(proxy, group_id):
        """
        Get the area ID with the given group ID using the Evident Microscope API. The group is assumed to only have one area and no compression level is used.
        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            group_id: ID of the group to get the area ID of.

        Returns:
            tuple: (EvidentError, area ID)
        """
        input_val = {"groupId": group_id, "level": 0, "area": 0}
        result = proxy.IDA.getArea(input_val)
        err = get_evident_error(result)
        if err.is_error:
            return err, None
        return err, result["areaId"]

    @staticmethod
    def get_channel_list(proxy, area_id):
        """
        Get the list of enabled channels for the given area ID using the Evident Microscope API.

        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            area_id: ID of the area to get the channel list of.
        Returns:
            tuple: (EvidentError, list of channel IDs)
        """
        input_val = {"areaId": area_id, "key": "EnabledChannelIdList"}
        result = proxy.IDA.getProperty(input_val)
        err = get_evident_error(result)
        if err.is_error:
            return err, None
        return err, result["value"]

    @staticmethod
    def get_image(proxy, filepath):
        """
        Get the image data and pixel sizes from the given file path using the Evident Microscope API. This function only works for an image with a single group and area.
        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            filepath: path to the file to get the image data from
        Returns:
            tuple: (image data as numpy array, pixel sizes as list of float)

        """
        inputVal = {"filename": filepath}
        result = proxy.IDA.open(inputVal)
        error = None

        # The function has a lot of nested ifs to correctly release all the IDs opens even in case of error.
        error_file = get_evident_error(result)
        if not error_file.is_error:
            fileId = result["fileId"]
            group_id = proxy.IDA.getGroup({"fileId": fileId, "group": 0})
            group_id_err = get_evident_error(group_id)
            if not group_id_err.is_error:
                area_id_err, area_id = EvidentCamera.get_area_id(proxy, group_id["groupId"])

                if not area_id_err.is_error:
                    pixel_sizes_err, pixel_sizes = EvidentCamera.get_pixel_sizes(proxy, area_id)

                    if not pixel_sizes_err.is_error:
                        channel_list_err, channel_list = EvidentCamera.get_channel_list(proxy, area_id)

                        if not channel_list_err.is_error:
                            api_axes = np.array(["TIMELAPSE", "ZSTACK", "LAMBDA"])
                            axes_exist_err, axes_exist, size_axes = EvidentCamera.get_axes_on_image(
                                proxy, area_id, api_axes
                            )

                            if not axes_exist_err.is_error:
                                multistack_img = None  # initilize as none because the width and height is not known yet
                                for iCh, iT, iZ, iL in itertools.product(
                                    np.arange(len(channel_list)),
                                    np.arange(size_axes[0]),
                                    np.arange(size_axes[1]),
                                    np.arange(size_axes[2]),
                                ):
                                    axes_of_img = api_axes[axes_exist]
                                    image_id_err, image_id = EvidentCamera.get_image_id(
                                        proxy,
                                        area_id,
                                        axes_of_img,
                                        np.array((iL, iZ, iT))[axes_exist],
                                        channel_list[iCh],
                                    )

                                    if not image_id_err.is_error:

                                        data_err, data = EvidentCamera.get_image_data(proxy, image_id)

                                        if data_err.is_error:
                                            error = data_err
                                        if multistack_img is None:
                                            multistack_img = np.zeros(
                                                (len(channel_list), size_axes[0], size_axes[1], size_axes[2])
                                                + data.shape,
                                                dtype=data.dtype,
                                            )
                                        multistack_img[iCh, iT, iZ, iL, :, :] = data

                                        proxy.IDA.releaseImage({"imageId": image_id[1]})
                                    else:
                                        error = image_id_err
                            else:
                                error = axes_exist_err
                        else:
                            error = channel_list_err
                    else:
                        error = pixel_sizes_err
                    proxy.IDA.releaseArea({"areaId": area_id})
                else:
                    error = area_id_err
                proxy.IDA.releaseGroup({"groupId": group_id})
            else:
                error = group_id_err
            proxy.IDA.close({"fileId": fileId})
        else:
            error = error_file
        if error is not None:
            error.throw_if_error()

        return multistack_img, pixel_sizes

    @staticmethod
    def get_pixel_sizes(proxy, area_id):
        """
        Get the pixel sizes for the given area ID using the Evident Microscope API. Example taken for RDKObj sample program from Evident.
        Args:
            proxy: xmlrpc client proxy to communicate with Evident microscope
            area_id: ID of the area to get the pixel sizes of.
        Returns:
            tuple: (EvidentError, list of pixel sizes as float (value in um))
        """

        inputVal = {"areaId": area_id, "key": "PixelLength"}
        result = proxy.IDA.getProperty(inputVal)
        err = get_evident_error(result)
        return err, result["value"]

    def busy(self) -> bool:
        """
        Returns true if the device is measuring else false.
        If get protocol progress returns a state other than IDLING, the device is busy. If the device is in WAITING state, it will stop the imaging to avoid getting stuck in that state.
        """
        # print(" busy function is used in EvidentCamera to check if the camera is measuring.")
        progress = self.get_imaging_progress()["state"]
        if progress == "WAITING":
            self.stop_imaging()
            progress = self.get_imaging_progress()["state"]
        return progress != "IDLING"

    @property  # this works as long as size of the image is set to 1024 x 1024 and 2 channels in the microscope software
    def data_shape(self) -> tuple[int, int]:
        """
        Get the shape of the acquired image data (height, width). Set to None as the value is dynamic.
        """
        return None

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

    def stop_imaging(self, execution_type: str = "EXECUTION_TYPE_MANUAL_MAIN"):
        """
        Stop imaging on the microscope.

        Args:
            execution_type (str): Type of execution. Options are "EXECUTION_TYPE_MANUAL_MAIN" or "EXECUTION_TYPE_MATL".
                                  Default is "EXECUTION_TYPE_MANUAL_MAIN".

        Returns:
            dict: Result dictionary containing result code, wait time, and target name.
        """
        inputVal = {"executionType": execution_type}

        out = self._proxy.Protocol.stopProtocol(inputVal)
        if out["result"] not in ("UNAVAILABLE", "OK"):
            err = get_evident_error(out)
            err.throw_if_error()
        return None


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
        self._moving = False  # Stores if a movement command has been issue and finished. This is for openwfs to not think that the stage is still moving when a movement was mad using the manual control of the microscope (because the definition of stop is only based on current position and target position).
        self._step = 100 * u.nm  # rounding resolution

    def busy(self) -> bool:
        """
        Returns true if the device is moving else false.
        If stage position is not equal to target position, the device is busy.
        """
        time.sleep(0.05)  # to avoid spamming the proxy with requests

        if not self._moving:
            return False

        arrive_at_target = u.allclose(self.xy, self._target_xy, atol=self._step / 2)
        self._moving = not arrive_at_target
        return self._moving

    @property
    def xy(self) -> tuple[Quantity[u.nm], Quantity[u.nm]]:
        # get current xy stage position
        inpuVal = {"settingId": "XY_STAGE_POSITION_SETTING"}
        result = self._proxy.Parameter.getParameter(inpuVal)

        error = get_evident_error(result)
        error.throw_if_error()

        stage_x = result["x"]
        stage_y = result["y"]
        return stage_x * u.nm, stage_y * u.nm

    @xy.setter
    def xy(self, value: tuple[Quantity, Quantity]):
        super()._start()  # wait for all previous commands to finish
        x, y = value
        # print(x)
        # print(self._step)
        x_rounded = round_quantity(x, self._step)
        y_rounded = round_quantity(y, self._step)
        self._target_xy = (round(x_rounded.to(u.nm)), round(y_rounded.to(u.nm)))

        x = round(x_rounded.to(u.nm).value)
        y = round(y_rounded.to(u.nm).value)
        inputVal = {"settingId": "XY_STAGE_POSITION_SETTING", "x": x, "y": y, "escapeEnabled": False}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        self._moving = True

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


class EvidentZStage(Actuator):
    """
    Z Stage in Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope

    Returns:
        EvidentZStage object with .position property to get and set the stage position.

    Example:
        stage = EvidentZStage(proxy)
        stage.z = 10 * u.nm # move stage to 10 in nm
        z = stage.z # get current stage position
    """

    def __init__(self, proxy):
        Actuator.__init__(
            self, duration=np.inf * u.ms, latency=0 * u.ms
        )  # duration is infinite since we don't know when the stage will reach the target position, busy() method will handle that.
        self._proxy = proxy
        self._target_z = None  # target position to move to (used in busy() method), in nm as int
        self._moving = False  # Stores if a movement command has been issue and finished. This is for openwfs to not think that the stage is still moving when a movement was mad using the manual control of the microscope (because the definition of stop is only based on current position and target position).
        self._step = 10 * u.nm

    def busy(self) -> bool:
        """
        Returns true if the device is moving else false.
        If stage position is not equal to target position, the device is busy.
        """
        time.sleep(0.05)  # to avoid spamming the proxy with requests
        # The stage has precision of 10nm so we consider the stage to be at the target position when within 10nm of the goal

        if not self._moving:
            return False

        arrive_at_target = np.abs(self.z - self._target_z) < self._step
        self._moving = not arrive_at_target
        return self._moving

    @property
    def position(self) -> Quantity[u.nm]:
        """
        Get current z stage position
        """
        inputVal = {"settingId": "Z_STAGE_POSITION_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)
        pos = result["stagePosition"]

        error = get_evident_error(result)
        error.throw_if_error()

        return -pos * u.nm

    @position.setter
    def position(self, z: Quantity):
        """
        Set z stage position
        """
        super()._start()
        self._target_z = z
        # Invert z because z should be positive when moving along the optical axis
        # The Z of the microscope is defined based on the objective lens
        z = -z
        z = int(z.to(u.nm).value)
        inputVal = {"settingId": "Z_STAGE_POSITION_SETTING", "stagePosition": z}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()
        self._moving = True

    @property
    def z(self) -> Quantity[u.nm]:
        """
        Get current z stage position
        """
        return self.position

    @z.setter
    def z(self, z: Quantity):
        """
        Set z stage position
        """
        self.position = z


class EvidentMicroscope:  # (Processor): How to calculate duration and latency, since it depend on both camera and stage?
    """
    Evident Microscope

    Arg:
        proxy: xmlrpc client proxy to communicate with Evident microscope

    Returns:
        EvidentMicroscope object with .camera and .stage properties.

    Example:
        mic = EvidentMicroscope(proxy)
        cam = mic.camera
        stage = mic.stage
    """

    def __init__(self, url):
        proxy = xmlrpc.client.ServerProxy(url)
        self._proxy = proxy
        self.cam = EvidentCamera(proxy)
        self.xy_stage = EvidentXYStage(proxy)
        self.z_stage = EvidentZStage(proxy)
        # super().__init__(self.cam, self.xy_stage)

    @property
    def version(self) -> str:
        """
        Get the product version.

        Returns:
            str: Version string, e.g., "1.1.1.1"
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_VERSION_SETTING"}
        return self._proxy.Parameter.getParameter(inputVal)["version"]

    def read(self):
        return self.cam.read()

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
        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def t_stack_step(self):
        inputVal = {"settingId": "LSM_TIME_LAPS_INTERVAL_SETTING"}
        result = self._proxy.Parameter.setParameter(inputVal)
        error = get_evident_error(result)
        error.throw_if_error()

        return result["interval"] * 10 ** (-result["unitOfInterval"])

    @t_stack_step.setter
    def t_stack_step(self, value):

        int_val = round(value.to(u.ms).value)
        inputVal = {"settingId": "LSM_TIME_LAPS_INTERVAL_SETTING", "interval": int_val, "unitOfInterval": 3}
        result = self._proxy.Parameter.setParameter(inputVal)
        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def t_stack_slice_num(self):
        inputVal = {"settingId": "LSM_TIME_LAPS_REPEAT_TIMES_SETTING"}
        result = self._proxy.Parameter.setParameter(inputVal)
        error = get_evident_error(result)
        error.throw_if_error()

        return result["repeadCount"]

    @t_stack_slice_num.setter
    def t_stack_slice_num(self, value: int):
        inputVal = {"settingId": "LSM_TIME_LAPS_REPEAT_TIMES_SETTING", "repeatCount": value}
        result = self._proxy.Parameter.setParameter(inputVal)
        error = get_evident_error(result)
        error.throw_if_error()

    def load_scan_parameters(self, file_path: str):
        """
        Load scan parameters from the specified file path.

        Args:
            file_path (str): File path to the target image for loading.

        Returns:
            dict: file ID.

        """

        if os.path.isfile(file_path):
            inputVal = {"targetName": file_path}
            result = self._proxy.Parameter.loadParameter(inputVal)
            error = get_evident_error(result)
            error.throw_if_error()
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_channel_list(self) -> list[str]:
        """
        Get the list of available channels.

        Returns:
            list[str]: List of channel Ids.
        """
        inputVal = {"settingId": "LSM_CHANNEL_LIST_SETTING"}
        out = self._proxy.Parameter.getParameter(inputVal)
        error = get_evident_error(out)
        error.throw_if_error()
        return out["channelList"]

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
        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["laserList"]

    def get_channel_laser(self, channel_id: str) -> list[str]:
        """
        Get the Laser set to Channel

        Args:
            channel_id (str): Channel Id acquired by LSM_CHANNEL_LIST_SETTING.

        Returns:
            list[str]: List of laser settings for the specified channel.
        """
        inputVal = {"settingId": "LSM_CHANNEL_LASER_SETTING", "channelId": channel_id}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["laser"]["laserId"]

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def objective(self) -> str:
        """
        Get the current objective lens name.

        Returns:
            str: Name of the current objective lens.
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_CURRENT_SETTING"}
        # print("The current objective lens is:", result["revolver"]["lensName"], "with magnification of:", result["revolver"]["magnification"])
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["revolver"]["lensName"]

    @objective.setter
    def objective(self, name: str):
        """
        Set the objective lens by specifying the revolver index.

        Args:
            revolver_index (int): Index of the nosepiece to be changed.
        """
        obj_list = self.objective_list
        obj_names = [obj["lensName"] for obj in obj_list]
        if name not in obj_names:
            raise ValueError(f"Objective lens '{name}' not found. Available lenses: {obj_names}")

        revolver_index = obj_list[obj_names.index(name)]["revolverIndex"]

        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_CURRENT_SETTING", "revolverIndex": revolver_index}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def objective_list(self) -> list[str]:
        """
        Get the list of available objective lenses.

        Returns:
            list[str]: List of objective lens names.
        """
        inputVal = {"settingId": "LSM_MICROSCOPE_OBJECTIVE_LENS_LIST_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["revolvers"]

    @property
    def z_stack_enabled(self) -> bool:
        """
        Get Enable/Disable status of Z stack

        Returns:
            bool: True if Z stack is enabled, False if disabled.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_ENABLE_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def z_stack_slice_num(self) -> int:
        """
        Get the number of slices of Z stack.

        Returns:
            int: Number of slices.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_SLICE_NUM_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

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

        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def z_stack_step_size(self) -> int:
        """
        Get the step size of Z stack.

        Returns:
            int: Step size in nm.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_STEP_SIZE_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["stepSize"] * u.nm

    @z_stack_step_size.setter
    def z_stack_step_size(self, z: int):
        """
        Set the step size of Z stack.

        Args:
            step_size (Quantity): Step size in nm to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_STEP_SIZE_SETTING", "stepSize": self.round2nm(z)}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

    @staticmethod
    def round2nm(val):
        return int(np.round(val.to(u.nm).value))

    @property
    def z_stack_end(self) -> int:
        """
        Get the step size of Z stack.

        Returns:
            int: Step size in nm.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_END_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["endPosition"] * u.nm

    @z_stack_end.setter
    def z_stack_end(self, z: int):
        """
        Set the end z of Z stack.

        Args:
            x (Quantity): Step size in nm to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_END_SETTING", "endPosition": self.round2nm(z)}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

    @property
    def z_stack_start(self):
        """
        Get the start of Z stack.

        Returns:
            int: start of z stack
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_START_SETTING"}
        result = self._proxy.Parameter.getParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

        return result["startPosition"] * u.nm

    @z_stack_start.setter
    def z_stack_start(self, z):
        """
        Set the end z of Z stack.

        Args:
            x (Quantity): start in nm to set.

        Returns:
            dict: Result dictionary containing result code.
        """
        inputVal = {"settingId": "LSM_Z_COORDINATE_START_SETTING", "startPosition": self.round2nm(z)}
        result = self._proxy.Parameter.setParameter(inputVal)

        error = get_evident_error(result)
        error.throw_if_error()

    def z_stack_read(self, z):
        """
        Read Z stack images at the specified z positions.

        Args:
            z (array-like): Array of z positions to read.

        Returns:
            numpy.ndarray: Z stack images acquired at the specified z positions
        """

        if z.size < 2:
            raise ValueError("Z must be a vector with size bigger or equal than 2")

        step = z[1] - z[0]
        if not np.allclose(np.diff(z), step):
            raise ValueError("The spacing of Z must be uniform")

        self.z_stack_end = z[-1]
        self.z_stack_start = z[0]
        self.z_stack_step_size = step
        self.z_stack_enabled = True

        img = self.cam.read()
        return img

    def _default_z_stack_read(self, z):
        """
        Default z stack read that does not use the z stack function. Temporary function which should be moved to the abstract microscope interface

        """

        self.z_stage.position = z[0]
        first_img = self.cam.read()  # read the first image to get the shape and dtype

        imgs = np.zeros((z.size,) + first_img.shape, dtype=first_img.dtype)
        imgs[0, ...] = first_img
        for i, z_pos in enumerate(z[1:], 1):
            self.z_stage.position = z_pos
            imgs[i, ...] = self.cam.read()

        return imgs

    def t_stack_read(self, t):
        if t.size < 2:
            raise ValueError("T must be a vector with more than 1 element")

        step = t[1] - t[0]

        if not np.allclose(np.diff(t), step):
            raise ValueError("The spacing of T must be uniform")

        self.t_stack_slice_num = t.size
        self.t_stack_step = step
        self.t_stack_enabled = True

        self.cam.read()

    def imaging_progress(self):
        return self.cam.get_imaging_progress()
