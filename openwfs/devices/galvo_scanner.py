from typing import Union
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from ..core import Detector
import nidaqmx as ni
from nidaqmx.constants import TaskMode

from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogUnscaledReader
from nidaqmx.constants import Edge, TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter


class LaserScanning(Detector):

    # TODO: more descriptive names for 'input_mapping' 'x_mirror_mapping'->'x_channel'
    # TODO: docstring
    # TODO: Add padding of sides: the image needs to be the same size, but it just needs to scan a larger area and
    # throw parts of it away during reconstructions
    # TODO: make the duration always correct, and the pixeldwelltime change with changing settings, then let setting
    # the pixel dwell time command set the duration to the correct value.
    def __init__(self, data_shape, pixel_size, input_mapping: str,
                 x_mirror_mapping: str, y_mirror_mapping: str, sample_rate, voltage_mapping, pos=(0, 0), input_min=-1,
                 input_max=1, delay=0,
                 duration: Quantity[u.ms] = 600 * u.ms, zoom=1, scan_padding=0, bidirectional=True, invert=True):
        super().__init__(data_shape=data_shape, pixel_size=pixel_size, duration=duration)

        self._resized = True
        self._image = None
        self._left = pos[0]
        self._top = pos[1]

        # NIDAQ device names
        self._input_mapping = input_mapping
        self._x_mirror_mapping = x_mirror_mapping
        self._y_mirror_mapping = y_mirror_mapping

        # Voltage limits scanning
        self._input_min = input_min
        self._input_max = input_max
        self._voltage_mapping = voltage_mapping

        # Reader voltage limits
        self._reader_min = -1
        self._reader_max = 1

        # Scan settings
        self._sample_rate = sample_rate
        self._delay = delay
        self._zoom = zoom
        self._scan_padding = scan_padding
        self._bidirectional = bidirectional

        self._invert = invert

    def _generate_scan_pattern(self):
        """
        Generates 2 numpy arrays to be used as input for the Galvo scanners,
        incorporating bidirectional scanning, padding, and an optional delay.

        The pattern length is determined by the duration and signal rate,
        independent of the dwell time. Padding and delay adjustments are included.
        """
        # Signal length based on duration and sample rate
        signal_length = int(self.duration.to(u.s) * self._sample_rate)

        min_v_x = ((-self.extent[0] / 2) / (self._voltage_mapping))
        max_v_x = ((self.extent[0] / 2) / (self._voltage_mapping))

        min_v_y = ((-self.extent[1] / 2) / (self._voltage_mapping))
        max_v_y = ((self.extent[1] / 2) / (self._voltage_mapping))

        # Create linear ranges for x and y axes
        x_step = (max_v_x - min_v_x) / (signal_length / self.data_shape[0] - 1)
        y_step = (max_v_y - min_v_y) / (self.data_shape[0] - 1)

        # Create ranges for x and y axes
        x_range = np.arange(min_v_x.value, max_v_x.value + x_step.value, x_step.value)
        y_range = np.arange(min_v_y.value, max_v_y.value + y_step.value, y_step.value)

        # Generate x and y steps using meshgrid
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        if self._bidirectional:
            x_grid[1::2] = np.flip(x_grid[1::2], axis=1)

        # Flatten and repeat the pattern to match the signal length
        x_steps = x_grid.flatten()
        y_steps = y_grid.flatten()

        # Add delay at the end of the pattern
        delay_length = int(self._delay.to(u.s) * self._sample_rate)
        x_steps = np.concatenate((x_steps, np.ones(delay_length) * x_steps[-1]))
        y_steps = np.concatenate((y_steps, np.ones(delay_length) * y_steps[-1]))

        return np.stack([x_steps, y_steps])

    def _setup_and_trigger_tasks(self, scanpattern):
        """Sets up and triggers the tasks for data acquisition."""
        self._number_of_samples = scanpattern[0].shape[0]

        self._write_task = ni.Task()
        self._read_task = ni.Task()
        self._sample_clk_task = ni.Task()

        # Configure the sample clock task
        self._sample_clk_task.co_channels.add_co_pulse_chan_freq(
            f"{self._input_mapping.split('/')[0]}/ctr0", freq=self._sample_rate.to(1 / u.s).value
        )
        self._sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=self._number_of_samples)

        samp_clk_terminal = f"/{self._input_mapping.split('/')[0]}/Ctr0InternalOutput"

        # Configure the analog output task
        ao_args = {'min_val': self._input_min.to(u.V).value,
                   'max_val': self._input_max.to(u.V).value}
        self._write_task.ao_channels.add_ao_voltage_chan(self._x_mirror_mapping, **ao_args)
        self._write_task.ao_channels.add_ao_voltage_chan(self._y_mirror_mapping, **ao_args)
        self._write_task.timing.cfg_samp_clk_timing(
            self._sample_rate.to(1 / u.s).value, source=samp_clk_terminal,
            active_edge=Edge.RISING, samps_per_chan=self._number_of_samples
        )

        # Configure the analog input task
        ai_args = {'min_val': self._reader_min,
                   'max_val': self._reader_max,
                   'terminal_config': TerminalConfiguration.RSE}
        self._read_task.ai_channels.add_ai_voltage_chan(self._input_mapping, **ai_args)
        self._read_task.timing.cfg_samp_clk_timing(
            self._sample_rate.to(1 / u.s).value, source=samp_clk_terminal,
            active_edge=Edge.FALLING, samps_per_chan=self._number_of_samples
        )

        # Write to the output task
        writer = AnalogMultiChannelWriter(self._write_task.out_stream)
        writer.write_many_sample(scanpattern)

        # Start the tasks
        self._read_task.start()
        self._write_task.start()
        self._sample_clk_task.start()

    def _read_acquired_data(self):
        """Reads the acquired data from the input task."""
        if not hasattr(self, '_read_task'):
            raise RuntimeError("Tasks have not been started. Call setup_and_trigger_tasks first.")

        reader = ni.stream_readers.AnalogUnscaledReader(self._read_task.in_stream)
        raw_data = np.zeros([1, self._number_of_samples], dtype=np.int16)
        reader.read_int16(raw_data, number_of_samples_per_channel=self._number_of_samples,
                          timeout=ni.constants.WAIT_INFINITELY)

        return raw_data

    def pmt_to_image(self, data):
        """
        Converts 1D PMT signal to a 2D image with binning.

        Parameters:
        data (numpy.ndarray): 1D array of PMT data.

        Returns:
        numpy.ndarray: 2D reconstructed image.

        ToDo: Padding needs to be analysed here.
        """
        # Skip the initial delay points
        delay_points = int(self._delay.to(u.s).value * self._sample_rate.value)
        data = data[delay_points:]

        # Calculate the number of points per line (x-axis) in the raw data
        raw_points_per_line = int(self.duration.to(u.s).value * self._sample_rate.value / self.data_shape[0])

        # Reshape the raw data into 2D
        reshaped_data = data.reshape((self.data_shape[0], raw_points_per_line))

        # Apply binning to match the data_shape
        binned_data = np.zeros(self.data_shape, dtype=np.uint16)
        bin_size = raw_points_per_line // self.data_shape[1]
        for i in range(self.data_shape[0]):
            for j in range(self.data_shape[1]):
                start_index = j * bin_size
                end_index = start_index + bin_size
                binned_data[i, j] = np.mean(reshaped_data[i, start_index:end_index])

        # If bidirectional, reverse every other line
        if self._bidirectional:
            binned_data[1::2] = binned_data[1::2, ::-1]

        return binned_data

    def _do_trigger(self):

        pattern = self._generate_scan_pattern()
        self._setup_and_trigger_tasks(pattern)

    def _fetch(self, out: Union[np.ndarray, None]) -> np.ndarray:

        raw_data = self._read_acquired_data()

        if out is None:
            out = self.pmt_to_image(raw_data.flatten())
        else:
            out[...] = self.pmt_to_image(raw_data.flatten())
        return out

    @property
    def left(self) -> int:
        return self._top

    @left.setter
    def left(self, value: int):
        self._top = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._data_shape = (value, self.data_shape[1])

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._data_shape = (self.data_shape[0], value)

    @property
    def dwell_time(self) -> Quantity[u.us]:
        return (self._duration / (self.data_shape[0] * self.data_shape[1])).to(u.us)

    @dwell_time.setter
    def dwell_time(self, value):
        self._duration = (value * (self.data_shape[0] * self.data_shape[1])).to(u.us)

    @property
    def duration(self) -> Quantity[u.ms]:
        return self._duration.to(u.ms)

    @duration.setter
    def duration(self, value):
        self._duration = value.to(u.ms)

    @property
    def delay(self) -> int:
        return self._delay  # add unit

    @delay.setter
    def delay(self, value: int):
        self._delay = value

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = value
        self._update_pixel_size()

    # Check if this influences dwelltime
    @property
    def scan_padding(self) -> float:
        return self._scan_padding

    @scan_padding.setter
    def scan_padding(self, value: float):
        self._scan_padding = value

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value: bool):
        self._bidirectional = value

    @property
    def invert(self) -> bool:
        return self._invert

    @invert.setter
    def invert(self, value: bool):
        self._invert = value
