from ..core import Actuator
import numpy as np
import astropy.units as u
from . import _MockModule
import time
import clr
import os
from concurrent.futures import Future, ThreadPoolExecutor

clr_loaded = type(clr) is not _MockModule

if clr_loaded:
    th_ine_files = [
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll",
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll",
        r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.InertialMotorCLI.dll"
    ]
    th_ine_lib_found = all(map(os.path.isfile, th_ine_files))
    if th_ine_lib_found:
        for f in th_ine_files:
            clr.AddReference(f)

        from Thorlabs.MotionControl.DeviceManagerCLI import (
            DeviceManagerCLI,
            SimulationManager
        )
        from Thorlabs.MotionControl.KCube.InertialMotorCLI import (
                KCubeInertialMotor,
                ThorlabsInertialMotorSettings,
                InertialMotorStatus
        )
        from System import Int32, UInt32


class KCubeInertial(Actuator):

    def __init__(self, serial_number: str, mock: bool = False, pair_channels: bool = False, timeout: u.Quantity = 20*u.s):
        super().__init__(duration = np.inf * u.ms, latency = 0 * u.ms)

        if not th_ine_lib_found:
            if not clr_loaded:
                raise ImportError("pythonnet (clr) is not installed. Please install pythonnet to use Thorlabs Kinesis libraries.")
            raise ImportError("Thorlabs Kinesis libraries not found. Please install Thorlabs Kinesis software.")

        if mock:
            SimulationManager.Instance.InitializeSimulations()
            print("Initialized Thorlabs Kinesis simulations.")


        DeviceManagerCLI.BuildDeviceList()

        serial_number_list = list(map(str, DeviceManagerCLI.GetDeviceList(Int32(97))))

        if serial_number not in serial_number_list and not mock:
            raise ValueError(f"Device with serial number {serial_number} not found. Available devices: {serial_number_list}")

        # create new device
        self.serial_number = str(serial_number)  # Serial number of device
        self.device = KCubeInertialMotor.CreateKCubeInertialMotor(self.serial_number)
        self.steprate = 500
        self.timeout = timeout
        # Connect
        self.device.Connect(self.serial_number)
        assert self.device.IsConnected

        time.sleep(0.25)
        
        # Ensure that the device settings have been initialized
        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(10000)  # 10 second timeout
            assert self.device.IsSettingsInitialized() is True

        # Start polling and enable channel
        self.device.StartPolling(250)  # 250ms polling rate
        time.sleep(0.25)
        self.device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable

        # Load any configuration settings needed by the controller/stage
        config = self.device.GetInertialMotorConfiguration(self.serial_number)
        settings = ThorlabsInertialMotorSettings.GetSettings(config)

        channels_array = []
        for ch_i in np.arange(4):
            th_name = f"Channel{ch_i + 1}"
            th_ch_i = getattr(InertialMotorStatus.MotorChannels, th_name)
            channels_array.append(th_ch_i)
            settings.Drive.Channel(th_ch_i).StepRate = self.steprate
            settings.Drive.Channel(th_ch_i).StepAcceleration = 1000

        self.channels_array = np.array(channels_array)
        self.device.SetSettings(settings, True, True)
        self.pair_channels(pair_channels)
        self.deviced_was_stoped = True
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._future = self._worker.submit(lambda: None)

    def __del__(self):
        self.device.StopPolling()
        self.device.Disconnect()

    def pair_channels(self, val):
        self.device.SetDualChannelMode(val)

    @property
    def position(self):
        out = np.zeros(self.channels_array.size, dtype=np.int32)
        for i, ch_i in enumerate(self.channels_array):
            out[i] = self.device.GetPosition(ch_i)

        self.deviced_was_stoped = False
        return out

    def throw_error_if_moving(self):
        assert not self.busy(), "Device is busy. Use self.wait() to wait for the device to finish moving or use self.stop() to stop the device before starting a new movement."
    
    def _move_to(self, *args_, **kwargs_):
        arr = args_[0]
        dists = np.abs(self.position - arr)

        if self.pair_channels:
            for i in np.array([0, 2]):
                if dists[i] > dists[i+1]:
                    i_long, i_short = i, i + 1
                else:
                    i_long, i_short = i + 1, i

                if dists[i_short] != 0:
                    self.device.MoveTo(self.channels_array[i_short], Int32(int(arr[i_short])), 0)

                if dists[i_long] != 0:
                    self.device.MoveTo(self.channels_array[i_long], Int32(int(arr[i_long])), int(self.timeout.to(u.ms).value))
                time.sleep(0.2)
        else:
            for i, ch_i in enumerate(self.channels_array):
                self.device.MoveTo(ch_i, Int32(int(arr[i])), self.timeout.to(u.ms).value)


    def _move_by(self, *args_, **kwargs_):
        arr = args_[0]
        dists = np.abs(arr)

        if self.pair_channels:
            for i in np.array([0, 2]):
                if dists[i] > dists[i+1]:
                    i_long, i_short = i, i + 1
                else:
                    i_long, i_short = i + 1, i

                if dists[i_short] != 0:
                    self.device.MoveBy(self.channels_array[i_short], Int32(int(arr[i_short])), 0)

                if dists[i_long] != 0:
                    self.device.MoveBy(self.channels_array[i_long], Int32(int(arr[i_long])), int(self.timeout.to(u.ms).value))
                time.sleep(0.2) # Need this, not sure why. Errors on position set otherwise
        else:
            for i, ch_i in enumerate(self.channels_array):
                self.device.MoveBy(ch_i, Int32(int(arr[i])), self.timeout.to(u.ms).value)

    @position.setter
    def position(self, arr):
        super()._start()
        assert arr.size == self.channels_array.size
        self.throw_error_if_moving()
        self._future = self._worker.submit(self._move_to, arr)
        print("Submitted move_to task")
    
    def move_by(self, deltas: np.ndarray):
        super()._start()
        assert deltas.size == self.channels_array.size
        self.throw_error_if_moving()
        self._future = self._worker.submit(self._move_by, deltas)

    def stop(self): 
        # Not sure that this can be done safely because device is used in another thread
        for ch_i in self.channels_array:
            self.device.Stop(ch_i)

    def busy(self):
        return not self._future.done()

