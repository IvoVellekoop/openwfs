import time

import astropy.units as u
from astropy.units import Quantity

from . import safe_import

ni = safe_import("nidaqmx", "nidaq")


class Gain:
    """
    A device that controls the voltage of a PMT gain using a NI data acquisition card.

    It is controlling a hamamatsu M9012 power supply, supplying power for a PMT H7422-40.

    It uses an analogue out connection to set the voltage,
    an analogue in channel to check the gain's self-protecting overload status,
    and a digital out that sends the signal to the gain to reset itself.

    It contains a boolean called reset that triggers the method on_reset.
    Ideally, we would like some button to execute methods in our devices,
    but this approach allows the user to execute methods in the device property manager.
    """

    def __init__(self, port_ao, port_ai, port_do, reset=False, gain=0 * u.V):
        """
        Initializes the Gain control.

        Args:
            port_ao (str): Port name for analog output (controls PMT voltage).
            port_ai (str): Port name for analog input (monitors overload status).
            port_do (str): Port name for digital output (controls gain reset).
            reset (bool): Initial state of the reset trigger.
            gain (Quantity[u.V]): Initial gain value.
        """
        self.port_ao = port_ao
        self.port_ai = port_ai
        self.port_do = port_do
        self._reset = reset
        self._gain = gain
        self.gain = gain  # triggers write to NI-DAQ

    def check_overload(self):
        with ni.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.port_ai)
            in_stream = task.in_stream

            data = in_stream.read(number_of_samples_per_channel=1)
            if data > 2.5:
                overload = True
            else:
                overload = False

            return overload

    def on_reset(self, value):
        if value:
            with ni.Task() as task:
                task.do_channels.add_do_chan(
                    self.port_do, line_grouping=nidaqmx.constants.LineGroupingLineGrouping.CHAN_FOR_ALL_LINES
                )
                task.write([True])
                time.sleep(1)
                task.write([False])

    @property
    def reset(self) -> bool:
        """Triggers the reset process if set to True."""
        return self._reset

    @reset.setter
    def reset(self, value: bool):
        self.on_reset(value)
        self._reset = value

    @property
    def gain(self) -> Quantity[u.V]:
        """Sets or gets the current gain value. Range: 0 to 0.9 volts."""
        # The range of values is the hardware supplier's defined voltage range. Setting the range here for safety
        return self._gain

    @gain.setter
    def gain(self, value: Quantity[u.V]):
        self._gain = value.to(u.V)
        with ni.Task() as write_task:
            channel = write_task.ao_channels.add_ao_voltage_chan(self.port_ao)
            channel.ao_min = 0
            channel.ao_max = 0.9
            write_task.write(self._gain.to_value(u.V))
