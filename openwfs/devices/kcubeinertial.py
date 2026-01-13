from ..core import Actuator
import numpy as np
import astropy.units as u
import time

try:
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
    clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.InertialMotorCLI.dll")
    import Thorlabs.MotionControl.DeviceManagerCLI as th_dev
    import Thorlabs.MotionControl.GenericMotorCLI as th_gen
    import Thorlabs.MotionControl.KCube.InertialMotorCLI as th_iner
except ImportError:
    th_lib_found = False
except Exception as e:
    print(f"Error importing Thorlabs Kinesis libraries: {e}")
    th_lib_found = False




class KCubeInertialMotor(Actuator):

    def __init__(self, serial_number: str, num_channels: int = 1, mock: bool = False):
        super().__init__(duration = np.inf * u.ms, latency = 0 * u.ms)

        if not th_lib_found:
            raise ImportError("Thorlabs Kinesis libraries not found. Please install Thorlabs Kinesis software.")

        if mock:
           th_dev.SimulationManager.Instance.InitializeSimulations()


        th_dev.BuildDeviceList()
        serial_number_list = th_dev.GetDeviceList(th_iner.DevicePrefix_KIM101);


        if serial_number not in serial_number_list:
            raise ValueError(f"Device with serial number {serial_number} not found. Available devices: {serial_number_list}")

        # create new device
        self.serial_number = str(serial_number)  # Serial number of device
        self.device = th_iner.KCubeInertialMotor.CreateKCubeInertialMotor(self.serial_number)
        self.last_movement = (0,0,0,0)
        self.current_position = (0,0,0,0)
        self.steprate = 500
        self.timeout = 100*u.s
        # Connect
        self.device.Connect(self.serial_number)
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
        settings = th_iner.ThorlabsInertialMotorSettings.GetSettings(config)

        for i in np.arange(num_channels):
            name = f"chan{i}"
            th_name = f"Channel{i+1}"
            setattr(self, name, getattr(th_iner.InertialMotorStatus.MotorChannels, th_name))
            channel = self.device.GetChannel(i)
            channel.LoadMotorConfiguration(settings.MotorConfiguration)
            settings.Drive.Channel(i).StepRate = self.steprate
            settings.Drive.Channel(i).StepAcceleration = 100000

        self.device.SetSettings(settings, True, True)

        for i in np.arange(num_channels):
            channel = getattr(self.device, f"chan{i}")
            channel.device.SetPositionAs(channel, 0)


        def __del__(self):
            self.device.StopPolling()
            self.device.Disconnect()

        def is_moving(self, axis: int) -> bool:
            chan = getattr(self.device, f"chan{axis}")
            return chan.IsMoving(chan)

        @property
        def position(self):
            out = np.zeros(self.num_channels, dtype=int)
            for i in np.arange(self.num_channels):
                chan = getattr(self.device, f"chan{i}")
                out[i] = chan.GetPosition(chan)
            return out

        @position.setter
        def position(self, arr):
            super()._start()
            assert arr.len == self.num_channels
            for i in np.arange(self.num_channels):
                chan = getattr(self.device, f"chan{i}")
                self.device.MoveTo(chan, arr[i], 0)

        def move_by(self, deltas: np.ndarray):
            super()._start()
            assert deltas.len == self.num_channels
            for i in np.arange(self.num_channels):
                chan = getattr(self.device, f"chan{i}")
                self.device.MoveBy(chan, deltas[i], 0)

        def busy(self):
            moving = False
            for i in np.arange(self.num_channels):
                moving = moving or self.is_moving(i)
            return moving













    


