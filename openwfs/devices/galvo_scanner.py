from typing import Union
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from ..core import Detector
import nidaqmx as ni
from nidaqmx.constants import TaskMode

from nidaqmx.constants import Edge, TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogUnscaledReader


# TODO: more descriptive names for 'input_mapping' 'x_mirror_mapping'->'x_channel'
# TODO: docstring
# TODO: Add padding of sides: the image needs to be the same size, but it just needs to scan a larger area and
# throw parts of it away during reconstructions
# TODO: make the duration always correct, and the pixeldwelltime change with changing settings, then let setting
# the pixel dwell time command set the duration to the correct value.


class LaserScanning(Detector):
    # LaserScanner(input=("ai/8", -1.0 * u.V, 1.0 * u.V))
    def __init__(self, data_shape,
                 input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 axis0: tuple[str, Quantity[u.V], Quantity[u.V]],
                 axis1: tuple[str, Quantity[u.V], Quantity[u.V]],
                 scale: Quantity[u.um / u.V],
                 sample_rate: Quantity[u.Hz],
                 delay=0.0,
                 zoom=1.0,
                 scan_padding=0, bidirectional=True, invert=True):
        """
        Args:
            data_shape (tuple[int,int]): number of data points (height, width) of the full field of view.
                Note that the ROI can be reduced later by setting width, height, top and left.
            input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for the input.
                 Tuple of: (name of the channel (e.g., 'ai/1'), minimum voltage, maximum voltage).
            axis0: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for controlling the axis 0 galvo (y-axis).
            axis1: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for controlling the axis 1 galvo (x-axis).
            scale (u.um / u.V):
                Conversion factor between voltage at the NiDaq card and displacement of the focus in the object plane.
                This may be an array of (height, width) conversion factors if the factors differ for the different axes.
                This factor is used to convert pixel positions `x_i, y_i` to voltages,
                 using the equation `(pos + [y_i, x_i]) * pixel_size / scale.
            sample_rate (u.Hz):
                Sample rate of the NiDaq input channel.
                The total scan time for a single frame will equal `data_shape[0] * data_shape[1] / sample_rate`
            delay:
            zoom:
            scan_padding:
            bidirectional:
            invert:
        """
        # settings for input/output channels
        self._in_channel = input[0]
        self._in_v_min = input[1]
        self._in_v_max = input[2]

        self._axis0_channel = axis0[0]
        self._axis0_v_min = axis0[1]
        self._axis0_v_max = axis0[2]

        self._axis1_channel = axis1[0]
        self._axis1_v_min = axis1[1]
        self._axis1_v_max = axis1[2]

        self._scale = scale.repeat(2) if scale.size == 1 else scale

        pixel_size = Quantity((
            (self._axis0_v_max - self._axis0_v_min) * self._scale[0] / (data_shape[0] - 1),
            (self._axis1_v_max - self._axis1_v_min) * self._scale[1] / (data_shape[1] - 1)
        ))
        duration = data_shape[0] * data_shape[1] / sample_rate

        super().__init__(data_shape=data_shape, pixel_size=pixel_size, duration=duration)

        self._resized = True  # indicate that the output signals need to be recomputed
        self._pos = (0, 0)  # left-top corner coordinates can be set through 'top' and 'left' to crop the ROI

        # Scan settings
        self._delay = delay
        self._zoom = zoom
        self._scan_padding = scan_padding
        self._bidirectional = bidirectional

        self._invert = invert
        self._write_task = None
        self._read_task = None
        self._clock_task = None

        self._valid = False  # indicates that `trigger()` should initialize the nidaq tasks and scan pattern
        self._scan_pattern = None

    def _setup_nidaq(self):
        """Sets up Nidaq task and i/o channels. Needs to be called every time a setting is changed"""
        sample_count = self.data_shape[0] * self.data_shape[1]
        sample_rate = (sample_count / self._duration).to_value(u.Hz)

        self._write_task = ni.Task()
        self._read_task = ni.Task()
        self._clock_task = ni.Task()

        # Configure the sample clock task
        self._clock_task.co_channels.add_co_pulse_chan_freq(
            f"{self._in_channel.split('/')[0]}/ctr0", freq=sample_rate
        )
        self._clock_task.timing.cfg_implicit_timing(samps_per_chan=sample_count)

        samp_clk_terminal = f"/{self._in_channel.split('/')[0]}/Ctr0InternalOutput"

        # Configure the analog output task
        self._write_task.ao_channels.add_ao_voltage_chan(self._axis0_channel,
                                                         min_val=self._axis0_v_min.to_value(u.V),
                                                         max_val=self._axis0_v_max.to_value(u.V))
        self._write_task.ao_channels.add_ao_voltage_chan(self._axis1_channel,
                                                         min_val=self._axis1_v_min.to_value(u.V),
                                                         max_val=self._axis1_v_max.to_value(u.V))
        self._write_task.timing.cfg_samp_clk_timing(
            sample_rate, source=samp_clk_terminal,
            active_edge=Edge.RISING, samps_per_chan=sample_count
        )

        # Configure the analog input task
        self._read_task.ai_channels.add_ai_voltage_chan(self._in_channel,
                                                        min_val=self._in_v_min.to_value(u.V),
                                                        max_val=self._in_v_max.to_value(u.V),
                                                        terminal_config=TerminalConfiguration.RSE)
        self._read_task.timing.cfg_samp_clk_timing(
            sample_rate, source=samp_clk_terminal,
            active_edge=Edge.FALLING, samps_per_chan=sample_count
        )

        self._writer = AnalogMultiChannelWriter(self._write_task.out_stream)
        self._reader = AnalogUnscaledReader(self._read_task.in_stream)

    def _generate_scan_pattern(self):
        """
        Generates array of voltages to be used as input for the Galvo scanners,
        incorporating bidirectional scanning, padding, and an optional delay.

        The pattern length is determined by the duration and signal rate,
        independent of the dwell time. Padding and delay adjustments are included.
        """

        # Create linear ranges for x and y axes
        # y_step = (self._axis0_v_max - self._axis0_v_min) / self.data_shape[0]
        # x_step = (self._axis1_v_max - self._axis1_v_min) / self.data_shape[1]

        y_range = np.linspace(self._axis0_v_max.to_value(u.V), self._axis0_v_min.to_value(u.V), self.data_shape[0])
        x_range = np.linspace(self._axis1_v_max.to_value(u.V), self._axis1_v_min.to_value(u.V), self.data_shape[1])

        # Generate x and y steps using meshgrid
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        if self._bidirectional:
            x_grid[1::2] = np.flip(x_grid[1::2], axis=1)

        # Flatten and repeat the pattern to match the signal length
        x_steps = x_grid.flatten()
        y_steps = y_grid.flatten()

        # Add delay at the end of the pattern
        # todo, this is broken, it jumps back to 1 volt instead of delaying the signal
        # delay_length = int(round(self._delay * self._sample_rate))
        # x_steps = np.concatenate((x_steps, np.ones(delay_length) * x_steps[-1]))
        # y_steps = np.concatenate((y_steps, np.ones(delay_length) * y_steps[-1]))

        return np.stack([x_steps, y_steps])

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
        # delay_points = int(self._delay.to(u.s).value * self._sample_rate.value)
        # data = data[delay_points:]
        #
        # # Calculate the number of points per line (x-axis) in the raw data
        # raw_points_per_line = int(self.duration.to(u.s).value * self._sample_rate.value / self.data_shape[0])
        #
        # # Reshape the raw data into 2D
        # reshaped_data = data.reshape((self.data_shape[0], raw_points_per_line))
        #
        # # Apply binning to match the data_shape
        # binned_data = np.zeros(self.data_shape, dtype=np.uint16)
        # bin_size = raw_points_per_line // self.data_shape[1]
        # for i in range(self.data_shape[0]):
        #     for j in range(self.data_shape[1]):
        #         start_index = j * bin_size
        #         end_index = start_index + bin_size
        #         binned_data[i, j] = np.mean(reshaped_data[i, start_index:end_index])
        #
        # # If bidirectional, reverse every other line
        # if self._bidirectional:
        #     binned_data[1::2] = binned_data[1::2, ::-1]
        #
        # return binned_data

    def _do_trigger(self):
        if not self._valid:
            self._scan_pattern = self._generate_scan_pattern()
            self._setup_nidaq()
            self._valid = True

        # write the samples to outpit in the x-y channels
        self._writer.write_many_sample(self._scan_pattern)

        # Start the tasks
        self._read_task.start()
        self._write_task.start()
        self._clock_task.start()

    def _fetch(self, out: Union[np.ndarray, None]) -> np.ndarray:  # noqa
        """Reads the acquired data from the input task."""
        sample_count = self.data_shape[0] * self.data_shape[1]
        if out is None:
            out = np.zeros(self.data_shape, dtype=np.int16)
        self._reader.read_int16(out.reshape(1,sample_count), number_of_samples_per_channel=sample_count,
                                timeout=ni.constants.WAIT_INFINITELY)

        if self._bidirectional:
            out[1::2, :] = out[1::2, ::-1]
        return out

    @property
    def left(self) -> int:
        return self._pos[1]

    @left.setter
    def left(self, value: int):
        self._pos = (self._pos[0], value)

    @property
    def top(self) -> int:
        return self._pos[0]

    @top.setter
    def top(self, value: int):
        self._pos = (value, self._pos[1])
        self._valid = False

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._data_shape = (value, self.data_shape[1])
        self._valid = False

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._data_shape = (self.data_shape[0], value)
        self._valid = False

    @property
    def dwell_time(self) -> Quantity[u.us]:
        return self._duration.to(u.us) / (self.data_shape[0] * self.data_shape[1])

    @dwell_time.setter
    def dwell_time(self, value: Quantity[u.us]):
        self._duration = value.to(u.us) * self.data_shape[0] * self.data_shape[1]
        self._valid = False

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
        self._valid = False

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = value
        self._valid = False

    # Check if this influences dwelltime
    @property
    def scan_padding(self) -> float:
        return self._scan_padding

    @scan_padding.setter
    def scan_padding(self, value: float):
        self._scan_padding = value
        self._valid = False

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value: bool):
        self._bidirectional = value
        self._valid = False

    @property
    def invert(self) -> bool:
        return self._invert

    @invert.setter
    def invert(self, value: bool):
        self._invert = value
        self._valid = False
