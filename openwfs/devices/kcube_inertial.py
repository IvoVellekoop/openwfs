from ..core import Actuator
import numpy as np
import astropy.units as u
import time
import clr
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# General notes about the implementation of this class:
# This code uses the DotNET interface from Thorlabs Kinesis to control
# the KCube KIM001 and KIM101. The code follows the OpenWFS interface to
# use the KCube with OpenWFS. The code uses a different proces (Thread)
# to correctly synchronise the movement: a separate process (thread) starts
# the movement of the stage using MoveTo or MoveBy (from Kinesis).
# This thread will be busy until the stage finishes the movement. Consequently,
# the busy function of OpenWFS can be defined as if the thread has or not finish.
# A consequence of the use of a separate thread is that the main
# process cannot try to communicate with the device while the move thread
# is communicating with the device. For this, function communicating with the
# device use the function KinesisHandler.throw_error_if_moving()


class KinesisHandler:
    """
    Class to handle the connection with the Kinesis software. This class is used to ensure that the Kinesis software is properly initialized and that the device list is built before trying to connect to a device. This is important because if the device list is not built, the code will not be able to find the device and will raise an error.
    """

    def __init__(self, kinesis_files):
        self.files = []
        self.add_files(kinesis_files)

    def add_files(self, kinesis_files):

        for f in kinesis_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(
                    f"Thorlabs Kinesis library file not found: {f}. Ensure that the correct path to the Kinesis libraries is provided and that the Kinesis software is installed. The library can be downloaded from https://www.thorlabs.com/kinesis-software."
                )

            clr.AddReference(f)
            self.files.append(f)

    @staticmethod
    def get_handler():
        global global_kinesis_handler
        if global_kinesis_handler is None:
            return KinesisHandler([])
        else:
            return global_kinesis_handler

    @staticmethod
    def disconnect(device):
        device.StopPolling()
        device.Disconnect()

    @staticmethod
    def connect(device):
        device.device.Connect(device.serial_number)
        if not device.device.IsConnected:
            raise ValueError(f"Failed to connect to device with serial number {device.serial_number}.")

        time.sleep(0.25)

        # Ensure that the device settings have been initialized
        if not device.device.IsSettingsInitialized():
            device.device.WaitForSettingsInitialized(10000)  # 10 second timeout
            if not device.device.IsSettingsInitialized():
                raise RuntimeError(
                    f"Device settings failed to initialize within timeout for device with serial number {device.serial_number}."
                )

        # Start polling and enable channel
        device.device.StartPolling(250)  # 250ms polling rate
        time.sleep(0.25)

        device.device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable

    @staticmethod
    def look_for_serialnumber(DeviceManagerCLI, device_codes, serial_number):
        DeviceManagerCLI.BuildDeviceList()

        serial_number_list = []
        for device_code in device_codes:
            serial_number_list.append(list(map(str, DeviceManagerCLI.GetDeviceList(device_code))))

        serial_number_list = [item for sublist in serial_number_list for item in sublist]

        if serial_number is None:
            if len(serial_number_list) == 1:
                serial_number = serial_number_list[0]
            elif len(serial_number_list) > 1:
                raise ValueError(
                    f"Multiple devices found. Please specify a serial number. Available devices: {serial_number_list}"
                )
            else:
                raise ValueError("No devices found.")

        if serial_number not in serial_number_list:
            raise ValueError(
                f"Device with serial number {serial_number} not found. Available devices: {serial_number_list}"
            )

        return str(serial_number)

    @staticmethod
    def throw_error_if_moving(device):
        """
        Convenience function to throw an error if the device is moving or if communication thread is communicating with the device.
        """
        if device.busy():
            raise RuntimeError(
                "Device is busy. Use self.wait() to wait for the device to finish moving or use self.stop() to stop the device."
            )

    @staticmethod
    def disconnect(device):
        device.device.StopPolling()
        device.device.Disconnect()

    def _find_kinesis_dlls(required_dll_names, folder_path=None):
        """
        Find Kinesis DLL files in a folder.

        Arguments:
            folder_path: str or Path or None - Path to the Thorlabs Kinesis folder. If None,
                defaults to C:\Program Files\Thorlabs\Kinesis.
            required_dll_names: list of str - Names of required DLL files (e.g.,
                ['Thorlabs.MotionControl.DeviceManagerCLI.dll', ...])

        Returns:
            list of str - Full paths to found DLL files in the same order as required_dll_names

        Raises:
            FileNotFoundError - If folder does not exist or required DLLs are not found
        """
        if folder_path is None:
            folder_path = r"C:\Program Files\Thorlabs\Kinesis"

        folder = Path(folder_path)

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Thorlabs Kinesis folder not found: {folder_path}. Ensure that the correct path to "
                "the Kinesis installation is provided. The software can be downloaded from "
                "https://www.thorlabs.com/kinesis-software."
            )

        found_dlls = []
        missing_dlls = []

        for dll_name in required_dll_names:
            dll_path = folder / dll_name
            if dll_path.is_file():
                found_dlls.append(str(dll_path))
            else:
                missing_dlls.append(dll_name)

        if missing_dlls:
            missing_str = ", ".join(missing_dlls)
            raise FileNotFoundError(
                f"Required Thorlabs Kinesis DLL files not found in {folder_path}. "
                f"Missing files: {missing_str}. Ensure that the Kinesis software "
                "is installed in this location. The software can be downloaded from "
                "https://www.thorlabs.com/kinesis-software."
            )

        return found_dlls


global_kinesis_handler = None


class KCubeInertial(Actuator):
    """
    Class to control KCube KIM001 and KIM101 from Thorlabs. To use this class the thorlabs Kinesis software must be installed. The software can be downloaded from https://www.thorlabs.com/kinesis-software. The communication with Kinesis is done using pythonnet (clr) which needs to be installed in the python environment.

    Arguments:
        serial_number: str - Serial number of the device to control. If not provided, the code
            will try to find a single connected device. If multiple devices are connected,
            an error is raised.
        pair_channels: bool defining if the motor 1 and 2, and 3 and 4 should be paired
            (i.e. moving simultaneously). Only important for KIM101.
        timeout: Quantity [u.s] - Defines the timeout time for the stage when performing
            movement. Defaults to 20 seconds.
        kinesis_folder: str - Path to the Thorlabs Kinesis installation folder. If not provided,
            defaults to C:\Program Files\Thorlabs\Kinesis.
    """

    def __init__(
        self,
        serial_number: str = None,
        pair_channels: bool = False,
        timeout: u.Quantity = 20 * u.s,
        kinesis_folder: str = None,
    ):

        required_dlls = [
            "Thorlabs.MotionControl.DeviceManagerCLI.dll",
            "Thorlabs.MotionControl.GenericMotorCLI.dll",
            "Thorlabs.MotionControl.KCube.InertialMotorCLI.dll",
        ]
        kinesis_files = KinesisHandler._find_kinesis_dlls(required_dlls, kinesis_folder)

        kinesis_handler = KinesisHandler.get_handler()
        kinesis_handler.add_files(kinesis_files)

        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.InertialMotorCLI import (
            KCubeInertialMotor,
            ThorlabsInertialMotorSettings,
            InertialMotorStatus,
        )
        from System import Int32

        self.DeviceManagerCLI = DeviceManagerCLI
        self.KCubeInertialMotor = KCubeInertialMotor
        self.ThorlabsInertialMotorSettings = ThorlabsInertialMotorSettings
        self.InertialMotorStatus = InertialMotorStatus
        self.Int32 = Int32

        super().__init__(duration=np.inf * u.ms, latency=0 * u.ms)

        DeviceManagerCLI.BuildDeviceList()

        self.serial_number = KinesisHandler.look_for_serialnumber(
            DeviceManagerCLI,
            device_codes=[KCubeInertialMotor.DevicePrefix_KIM101, KCubeInertialMotor.DevicePrefix_KIM001],
            serial_number=serial_number,
        )

        self.device = KCubeInertialMotor.CreateKCubeInertialMotor(self.serial_number)
        self.timeout = timeout
        # Connect
        KinesisHandler.connect(self)

        self._worker = ThreadPoolExecutor(max_workers=1)
        self._future = self._worker.submit(lambda: None)

        num_channel = 1 if self.device.IsSingleChannelDevice() else 4

        # Load any configuration settings needed by the controller/stage
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = ThorlabsInertialMotorSettings.GetSettings(config)

        # Get all channels of the device in an array format
        channels_array = []
        for ch_i in np.arange(num_channel):
            th_name = f"Channel{ch_i + 1}"
            th_ch_i = getattr(InertialMotorStatus.MotorChannels, th_name)
            channels_array.append(th_ch_i)

        self.channels_array = np.array(channels_array)
        self.device.SetSettings(settings, True, True)
        self.pair_channels = pair_channels
        self._get_velocity_acceleration()

    def __del__(self):
        KinesisHandler.disconnect(self)

    @property
    def pair_channels(self) -> bool:
        """
        Get if the channels are paired (i.e. moving simultaneously)

        Returns:
            bool - True if the channels 1 and 2, and 3 and 4 are paired. False otherwise.
        """
        KinesisHandler.throw_error_if_moving(self)
        return self.device.IsDualChannelMode()

    @pair_channels.setter
    def pair_channels(self, val: bool) -> None:
        """
        Set if the channels should be paired (i.e. moving simultaneously)

        Arguments:
            val: bool - True to pair channels 1 with 2, and 3 with 4. False otherwise.
        """

        KinesisHandler.throw_error_if_moving(self)
        self.device.SetDualChannelMode(val)
        time.sleep(0.2)

    @property
    def velocity(self) -> u.Quantity[1 / u.s]:
        """
        Returns:
            np.array [1/u.s] - Velocity of the stage. The array has one element per channel.
        """
        vel, acc = self._get_velocity_acceleration()
        return vel

    @velocity.setter
    def velocity(self, val: u.Quantity[1 / u.s]):
        """
        Set the velocity of the stage in steps/s
        Arguments:
            val: np.array [1/u.s] - Velocity to set. The array has one element per channel.
        """
        self._set_velocity_acceleration(val, self._acceleration)

    @property
    def acceleration(self) -> u.Quantity[1 / u.s**2]:
        """
        Gets the acceleration of the stage in steps/s^2. This function will probe the device for the current acceleration.
        Returns:
            np.array [1/u.s**2] - Acceleration of the stage. The array has one element per channel.
        """
        vel, acc = self._get_velocity_acceleration()
        return acc

    @acceleration.setter
    def acceleration(self, val: u.Quantity[1 / u.s**2]):
        """
        Set the acceleration of the stage in steps/s^2
        Arguments:
            val: np.array [1/u.s**2] - Acceleration to set. The array has one element per channel.
        """
        self._set_velocity_acceleration(self._velocity, val)

    def _set_velocity_acceleration(self, velocity, acceleration):
        KinesisHandler.throw_error_if_moving(self)
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = self.ThorlabsInertialMotorSettings.GetSettings(config)
        for i, ch_i in enumerate(self.channels_array):
            settings.Drive.Channel(ch_i).StepRate = self.Int32(int(velocity[i].to(1 / u.s).value))
            settings.Drive.Channel(ch_i).StepAcceleration = self.Int32(int(acceleration[i].to(1 / u.s**2).value))
        self.device.SetSettings(settings, True, True)
        self._get_velocity_acceleration()

    def _get_velocity_acceleration(self):
        KinesisHandler.throw_error_if_moving(self)
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = self.ThorlabsInertialMotorSettings.GetSettings(config)
        vel = []
        acc = []
        for ch_i in self.channels_array:
            vel.append(settings.Drive.Channel(ch_i).StepRate)
            acc.append(settings.Drive.Channel(ch_i).StepAcceleration)
        self._velocity = np.array(vel) * (1 / u.s)
        self._acceleration = np.array(acc) * (1 / u.s**2)
        return self._velocity, self._acceleration

    @property
    def position(self) -> np.ndarray:
        KinesisHandler.throw_error_if_moving(self)
        out = np.zeros(self.channels_array.size, dtype=np.int32)
        for i, ch_i in enumerate(self.channels_array):
            out[i] = self.device.GetPosition(ch_i)

        return out

    @position.setter
    def position(self, arr: np.ndarray):
        """
            Moves the device to the specified absolute positions in steps.

        Arguments:
            arr: np.ndarray - Array with the absolute positions to move each channel
        """
        if not arr.size == self.channels_array.size:
            raise ValueError(
                f"Size of position array ({arr.size}) does not match number of channels ({self.channels_array.size})."
            )

        KinesisHandler.throw_error_if_moving(self)
        super()._start()
        self._future = self._worker.submit(self._move_to, arr, self.position, self.pair_channels, True)

    @staticmethod
    def movement_time(
        distance: int, velocity: u.Quantity[1 / u.s], acceleration: u.Quantity[1 / u.s**2]
    ) -> u.Quantity[u.s]:
        """
        Returns the time required to move a given distance with a given velocity and acceleration. This function assumes a trapezoidal velocity profile, which is the default for Kinesis. The function calculates the time required to accelerate to the velocity, the time required to decelerate from the velocity, and the time required to move at constant velocity. If the distance is too short to reach the velocity, the function calculates the time required to accelerate and decelerate without reaching the velocity.

        """
        distance_acceleration = velocity**2 / (2 * acceleration)
        ind_achieve_max_velocity = distance >= 2 * distance_acceleration
        time = 2 * np.sqrt(distance / acceleration)
        time[ind_achieve_max_velocity] = (
            2 * velocity[ind_achieve_max_velocity] / acceleration[ind_achieve_max_velocity]
            + (distance[ind_achieve_max_velocity] - 2 * distance_acceleration[ind_achieve_max_velocity])
            / velocity[ind_achieve_max_velocity]
        )
        return time

    def _move_to(self, arr, current_position, pair_channels, is_move_to):
        """
            Function to be ran by the thread to move the stage to an absolute position

        Arguments:
            arr: np.ndarray - Array with the absolute positions to move each channel
            current_position: np.ndarray - Array with the current positions of each channel
            pair_channels: bool - If the channels are paired
            is_move_to: bool - True if the movement is to the position and False if the movement is by a relative amount.
        """
        dists = np.abs(current_position - arr) if is_move_to else np.abs(arr)

        time_required = self.movement_time(dists, self._velocity, self._acceleration)

        api_move_function = self.device.MoveTo if is_move_to else self.device.MoveBy

        if pair_channels:
            # If the channels are paired, the code finds the motor from the paired channels travels
            # further. It starts the movement of that channel last to ensure that when the movement
            # finishes the other motor already finished.
            for i in np.array([0, 2]):
                if time_required[i] > time_required[i + 1]:
                    i_long, i_short = i, i + 1
                else:
                    i_long, i_short = i + 1, i

                if not np.isclose(dists[i_short], 0):  # Need to check this otherwise Kinesis will block the thread
                    api_move_function(self.channels_array[i_short], self.Int32(int(arr[i_short])), 0)

                if not np.isclose(dists[i_long], 0):
                    api_move_function(
                        self.channels_array[i_long], self.Int32(int(arr[i_long])), int(self.timeout.to(u.ms).value)
                    )
                    time.sleep(0.2)  # Need this, not sure why. Errors on position set otherwise
        else:
            for i, ch_i in enumerate(self.channels_array):
                if not np.isclose(dists[i], 0):
                    api_move_function(ch_i, self.Int32(int(arr[i])), int(self.timeout.to(u.ms).value))
                    time.sleep(0.2)

    def move_by(self, deltas: int) -> None:
        """
            Moves the device by the specified relative distances in steps.

        Arguments:
            deltas: np.ndarray - Array with the relative distances to move each channel
        """
        if not deltas.size == self.channels_array.size:
            raise ValueError(
                f"Size of deltas array ({deltas.size}) does not match number of channels ({self.channels_array.size})."
            )
        super()._start()
        KinesisHandler.throw_error_if_moving(self)
        self._future = self._worker.submit(self._move_to, deltas, self.position, self.pair_channels, False)

    def stop(self):
        """
        Stops the movement of the device. This function should only be used in an emergency
        because it stops the communication with the device in the middle of a move command.
        """
        # Avoid using this because it can stop a thread in the middle of a device command
        self._future.cancel()
        # Not sure that this can be done safely because device is used in another thread
        for ch_i in self.channels_array:
            self.device.Stop(ch_i)

    def busy(self):
        """
        Returns True if the device is currently moving or communicating with the device.
        """
        # This function works because the thread will be locked by kinesis while a movement
        # is ongoing.
        return not self._future.done()


class MotorizedFilterFlip(Actuator):
    """
    Class to control Motorized Filter Flip (MFF101) from Thorlabs. To use this class the thorlabs Kinesis software must be installed. The software can be downloaded from https://www.thorlabs.com/kinesis-software. The communication with Kinesis is done using pythonnet (clr) which needs to be installed in the python environment.

    Arguments:
        serial_number: str - Serial number of the device to control. If not provided, the code
            will try to find a single connected device. If multiple devices are connected,
            an error is raised.
        timeout: Quantity [u.s] - Defines the timeout time for the stage when performing
            movement. Defaults to 20 seconds.
        kinesis_folder: str - Path to the Thorlabs Kinesis installation folder. If not provided,
            defaults to C:\Program Files\Thorlabs\Kinesis.
    """

    def __init__(self, serial_number: str = None, timeout: u.Quantity = 20 * u.s, kinesis_folder: str = None):

        required_dlls = [
            "Thorlabs.MotionControl.DeviceManagerCLI.dll",
            "Thorlabs.MotionControl.GenericMotorCLI.dll",
            "Thorlabs.MotionControl.FilterFlipperCLI.dll",
        ]
        kinesis_files = KinesisHandler._find_kinesis_dlls(required_dlls, kinesis_folder)

        kinesis_handler = KinesisHandler.get_handler()
        kinesis_handler.add_files(kinesis_files)

        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.FilterFlipperCLI import (
            FilterFlipper,
        )
        from System import Int32

        self.DeviceManagerCLI = DeviceManagerCLI
        self.FilterFlipper = FilterFlipper
        self.Int32 = Int32

        super().__init__(duration=np.inf * u.ms, latency=0 * u.ms)

        # create new device
        self.serial_number = KinesisHandler.look_for_serialnumber(
            DeviceManagerCLI, device_codes=[FilterFlipper.DevicePrefix], serial_number=serial_number
        )
        self.device = FilterFlipper.CreateFilterFlipper(self.serial_number)
        self.timeout = timeout

        KinesisHandler.connect(self)

        self._worker = ThreadPoolExecutor(max_workers=1)
        self._future = self._worker.submit(lambda: None)

    def __del__(self):
        KinesisHandler.disconnect(self)

    def _move_to(self, pos):
        """
            Function to be ran by the thread to move the stage to an absolute position

        Arguments:
            pos: int - Absolute position to move the device to.
        """
        self.device.SetPosition(self.Int32(int(pos)), int(self.timeout.to(u.ms).value))

    def home(self):
        """
        Moves the device to the home position (0).
        """
        KinesisHandler.throw_error_if_moving(self)
        super()._start()
        self._future = self._worker.submit(self.device.Home, int(self.timeout.to(u.ms).value))

    @property
    def position(self) -> int:
        KinesisHandler.throw_error_if_moving(self)
        return self.device.GetPosition()

    @position.setter
    def position(self, pos: int):
        """
            Moves the device to the specified absolute position.

        Arguments:
            pos: int - Absolute position to move the device to.
        """
        KinesisHandler.throw_error_if_moving(self)
        super()._start()
        self._future = self._worker.submit(self._move_to, pos)

    def busy(self):
        """
        Returns True if the device is currently moving or communicating with the device.
        """
        # This function works because the thread will be locked by kinesis while a movement
        # is ongoing.
        return not self._future.done()
