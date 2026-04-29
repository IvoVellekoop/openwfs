from ..core import Actuator
import numpy as np
import astropy.units as u
from . import _MockModule
import time
import clr
import os
from concurrent.futures import ThreadPoolExecutor

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
# device use the function throw_error_if_moving()

clr_loaded = type(clr) is not _MockModule

if clr_loaded:
    th_ine_files = [
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll",
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll",
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.InertialMotorCLI.dll",
    ]
    th_ine_lib_found = all(map(os.path.isfile, th_ine_files))
    if th_ine_lib_found:
        for f in th_ine_files:
            clr.AddReference(f)

        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.InertialMotorCLI import (
            KCubeInertialMotor,
            ThorlabsInertialMotorSettings,
            InertialMotorStatus,
        )
        from System import Int32


class KCubeInertial(Actuator):
    """
    Class to control KCube KIM001 and KIM101 from Thorlabs.

    Arguments:
        serial_number: str - Serial number of the device to control. If not provided, the code
            will try to find a single connected device. If multiple devices are connected,
            an error is raised.
        pair_channels: bool defining if the motor 1 and 2, and 3 and 4 should be paired
            (i.e. moving simultaneously). Only important for KIM101.
        timeout: Quantity [u.s] - Defines the timeout time for the stage when performing
            movement. Defaults to 20 seconds.
    """

    def __init__(self, serial_number: str = "", pair_channels: bool = False, timeout: u.Quantity = 20 * u.s):
        super().__init__(duration=np.inf * u.ms, latency=0 * u.ms)

        if not clr_loaded:
            raise ImportError("pythonnet (clr) is not installed. Install pythonnet to use Thorlabs Kinesis libraries.")
        if not th_ine_lib_found:
            raise ImportError(
                "Thorlabs Kinesis libraries not found. Download and install Thorlabs Kinesis software from https://www.thorlabs.com/software-pages/motion_control/."
            )

        DeviceManagerCLI.BuildDeviceList()

        serial_number_list = list(map(str, DeviceManagerCLI.GetDeviceList(Int32(97))))

        if serial_number == "":
            if len(serial_number_list) == 1:
                serial_number = serial_number_list[0]
            elif len(serial_number_list) > 1:
                raise ValueError(
                    f"Multiple KCube Inertial Motor devices found. Please specify a serial number. Available devices: {serial_number_list}"
                )
            else:
                raise ValueError("No KCube Inertial Motor devices found.")

        if serial_number not in serial_number_list:
            raise ValueError(
                f"Device with serial number {serial_number} not found. Available devices: {serial_number_list}"
            )

        # create new device
        self.serial_number = str(serial_number)  # Serial number of device
        self.device = KCubeInertialMotor.CreateKCubeInertialMotor(self.serial_number)
        self.timeout = timeout
        # Connect
        self.device.Connect(self.serial_number)
        assert self.device.IsConnected

        time.sleep(0.25)

        # Ensure that the device settings have been initialized
        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(10000)  # 10 second timeout
            assert self.device.IsSettingsInitialized() is True

        self._worker = ThreadPoolExecutor(max_workers=1)
        self._future = self._worker.submit(lambda: None)

        # Start polling and enable channel
        self.device.StartPolling(250)  # 250ms polling rate
        time.sleep(0.25)
        self.device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable

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
        self.device.StopPolling()
        self.device.Disconnect()

    @property
    def pair_channels(self):
        """
        Get if the channels are paired (i.e. moving simultaneously)

        Returns:
            bool - True if the channels 1 and 2, and 3 and 4 are paired. False otherwise.
        """
        self.throw_error_if_moving()
        return self.device.IsDualChannelMode()

    @pair_channels.setter
    def pair_channels(self, val):
        """
        Set if the channels should be paired (i.e. moving simultaneously)

        Arguments:
            val: bool - True to pair channels 1 with 2, and 3 with 4. False otherwise.
        """

        self.throw_error_if_moving()
        self.device.SetDualChannelMode(val)
        time.sleep(0.2)

    @property
    def velocity(self):
        """
        Gets the velocity of the stage in steps/s. This function will probe the device for the
        current velocity.
        Returns:
            nd.array [1/u.s] - Velocity of the stage. The array has one element per channel.
        """
        vel, acc = self._get_velocity_acceleration()
        return vel

    @velocity.setter
    def velocity(self, val):
        """
        Set the velocity of the stage in steps/s
        Arguments:
            val: nd.array [1/u.s] - Velocity to set. The array has one element per channel.
        """
        self._set_velocity_acceralleration(val, self._acceleration)

    @property
    def acceleration(self):
        """
        Gets the acceleration of the stage in steps/s^2. This function will probe the device for the current acceleration.
        Returns:
            nd.array [1/u.s**2] - Acceleration of the stage. The array has one element per channel.
        """
        vel, acc = self._get_velocity_acceleration()
        return acc

    @acceleration.setter
    def acceleration(self, val):
        """
        Set the acceleration of the stage in steps/s^2
        Arguments:
            val: nd.array [1/u.s**2] - Acceleration to set. The array has one element per channel.
        """
        self._set_velocity_acceralleration(self._velocity, val)

    def _set_velocity_acceralleration(self, velocity, acceleration):
        self.throw_error_if_moving()
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = ThorlabsInertialMotorSettings.GetSettings(config)
        for i, ch_i in enumerate(self.channels_array):
            settings.Drive.Channel(ch_i).StepRate = Int32(int(velocity[i].to(1 / u.s).value))
            settings.Drive.Channel(ch_i).StepAcceleration = Int32(int(acceleration[i].to(1 / u.s**2).value))
        self.device.SetSettings(settings, True, True)
        self._get_velocity_acceleration()

    def _get_velocity_acceleration(self):
        self.throw_error_if_moving()
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = ThorlabsInertialMotorSettings.GetSettings(config)
        vel = []
        acc = []
        for ch_i in self.channels_array:
            vel.append(settings.Drive.Channel(ch_i).StepRate)
            acc.append(settings.Drive.Channel(ch_i).StepAcceleration)
        self._velocity = np.array(vel) * (1 / u.s)
        self._acceleration = np.array(acc) * (1 / u.s**2)
        return self._velocity, self._acceleration

    @property
    def position(self):
        self.throw_error_if_moving()
        out = np.zeros(self.channels_array.size, dtype=np.int32)
        for i, ch_i in enumerate(self.channels_array):
            out[i] = self.device.GetPosition(ch_i)

        return out

    def throw_error_if_moving(self):
        """
        Convenience function to throw an error if the device is moving or if communication thread is communicating with the device.
        """
        assert (
            not self.busy()
        ), "Device is busy. Use self.wait() to wait for the device to finish moving or use self.stop() to stop the device."

    def _move_to(self, *args_, **kwargs_):
        """
            Function to be ran by the thread to move the stage to an absolute position

        Arguments:
            args_[0]: np.ndarray - Array with the absolute positions to move each channel
            args_[1]: np.ndarray - Array with the current positions of each channel
            args_[2]: bool - If the channels are paired
        """
        arr = args_[0]
        current_position = args_[1]
        dists = np.abs(current_position - arr)
        pair_channels = args_[2]

        if pair_channels:
            # If the channels are paired, the code finds the motor from the paired channels travels
            # further. It starts the movement of that channel last to ensure that when the movement
            # finishes the other motor already finished.
            for i in np.array([0, 2]):
                if dists[i] > dists[i + 1]:
                    i_long, i_short = i, i + 1
                else:
                    i_long, i_short = i + 1, i

                if dists[i_short] != 0:  # Need to check this otherwise Kinesis will block the thread
                    self.device.MoveTo(self.channels_array[i_short], Int32(int(arr[i_short])), 0)

                if dists[i_long] != 0:
                    self.device.MoveTo(
                        self.channels_array[i_long], Int32(int(arr[i_long])), int(self.timeout.to(u.ms).value)
                    )
                    time.sleep(0.2)  # Need this, not sure why. Errors on position set otherwise
        else:
            for i, ch_i in enumerate(self.channels_array):
                if dists[i] != 0:
                    self.device.MoveTo(ch_i, Int32(int(arr[i])), int(self.timeout.to(u.ms).value))
                    time.sleep(0.2)

    def _move_by(self, *args_, **kwargs_):
        """
            Function to be ran by the thread to move the stage by a relative amount

        Arguments:
            args_[0]: np.ndarray - Array with the relative distances to move each channel
            args_[1]: bool - If the channels are paired
        """
        arr = args_[0]
        dists = np.abs(arr)
        pair_channels = args_[1]

        if pair_channels:
            # If the channels are paired, the code finds the motor from the paired channels travels
            # further. It starts the movement of that channel last to ensure that when then movement
            # finishes the other motor already finished.
            for i in np.array([0, 2]):
                if dists[i] > dists[i + 1]:
                    i_long, i_short = i, i + 1
                else:
                    i_long, i_short = i + 1, i

                if dists[i_short] != 0:  # Need to check this otherwise Kinesis will block the thread
                    self.device.MoveBy(self.channels_array[i_short], Int32(int(arr[i_short])), 0)

                if dists[i_long] != 0:
                    self.device.MoveBy(
                        self.channels_array[i_long], Int32(int(arr[i_long])), int(self.timeout.to(u.ms).value)
                    )
                    time.sleep(0.2)  # Need this, not sure why. Errors on position set otherwise
        else:
            for i, ch_i in enumerate(self.channels_array):
                if dists[i] != 0:
                    self.device.MoveBy(ch_i, Int32(int(arr[i])), int(self.timeout.to(u.ms).value))
                    time.sleep(0.2)

    @position.setter
    def position(self, arr):
        """
            Moves the device to the specified absolute positions in steps.

        Arguments:
            arr: np.ndarray - Array with the absolute positions to move each channel
        """
        assert arr.size == self.channels_array.size
        super()._start()
        self.throw_error_if_moving()
        self._future = self._worker.submit(self._move_to, arr, self.position, self.pair_channels)

    def move_by(self, deltas: np.ndarray):
        """
            Moves the device by the specified relative distances in steps.

        Arguments:
            deltas: np.ndarray - Array with the relative distances to move each channel
        """
        assert deltas.size == self.channels_array.size
        super()._start()
        self.throw_error_if_moving()
        self._future = self._worker.submit(self._move_by, deltas, self.pair_channels)

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
        # This function works because the thread will be lock by kinesis while a movement
        # is ongoing.
        return not self._future.done()
