from typing import Annotated
from typing import Union
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from ..core import Detector
from .Pyscanner import single_capture
import nidaqmx as ni
from nidaqmx.constants import TaskMode

from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogUnscaledReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

class LaserScanning(Detector):

    def __init__(self, left=0, top=0, data_shape = (100,100), input_mapping='Dev4/ai24',
                 x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3', full_scan_range = 884.4 * u.um, input_min=-1, input_max=1, delay=0,
                 duration: Quantity[u.ms] = 600 * u.ms, zoom=1, scan_padding=0, bidirectional=True, invert=True):
        super().__init__(data_shape=data_shape, pixel_size=1 * u.um, duration=duration)

        self._resized = True
        self._image = None
        self._left = left
        self._top = top

        # NIDAQ device names
        self._input_mapping = input_mapping
        self._x_mirror_mapping = x_mirror_mapping
        self._y_mirror_mapping = y_mirror_mapping

        # Voltage limits scanning
        self._input_min = input_min
        self._input_max = input_max
        self._full_scan_range = full_scan_range

        # Scan settings
        self._delay = delay
        self._zoom = zoom
        self._scan_padding = scan_padding
        self._bidirectional = bidirectional
        self._invert = invert


        self._sample_rate = self.calculate_sample_rate()

    def scanpattern(self):
        """This produces 2 numpy arrays which can be used as input for the Galvo scanners

        Todo: make padding structured (sinusoid instead of flat line)
        """

        # This is the linear signal. Everything after is padding & structuring. Adapt here for custom patterns.
        rangex = np.linspace(self._input_min, self._input_max, self.data_shape[1])
        rangey = np.linspace(self._input_min, self._input_max, self.data_shape[0])

        xsteps = np.array([])
        ysteps = np.array([])



        for ind, y in enumerate(rangey):
            if self._bidirectional:
                if (ind % 2) == 0:
                    # Add padding at the start of the row
                    xsteps = np.append(xsteps, np.append(np.ones(self._scan_padding) * rangex[0], rangex))
                else:
                    # Add padding at the start of the row, then flip for odd rows
                    xsteps = np.append(xsteps, np.append(np.ones(self._scan_padding) * rangex[-1], np.flip(rangex)))
            else:
                # Add padding for unidirectional scanning
                xsteps = np.append(xsteps, np.append(np.ones(self._scan_padding) * rangex[0], rangex))

        for y in rangey:
            # Repeat each y-value for the length of a padded x-row
            ysteps = np.append(ysteps, np.ones(len(rangex) + self._scan_padding) * y)

        # Reading and writing timepoints lag by 1 point: This corrects it
        xsteps = np.append(xsteps, xsteps[-1])
        ysteps = np.append(ysteps, ysteps[-1])

        return np.stack([xsteps, ysteps])

    def setup_reader_writer(self,scanpattern):
        """Function adapted from NI forum that has an electrical output signal and a electrical input signal.
        Because it goes into the dac, the input for the galvos is called output (analog out)
        and the output of the PMT is called input (analog in)

        It can handle only 1 in- and 2 outputs

        outdata: numpy array or list, will be output in analog out channel (V)

        sr: signal rate (/second), default is 500.000, the maximum of the NI USB-6341.
        Note; the NI PCIe-6363 in the lab has a maximum of 2.000.000, so this function can be overclocked.

        returns indata: measured signal from analog in channel (V)
        """
        # TODO: Make a buffer-loading & trigger function seperately
        # TODO: Make the function robust for different channel numbers

        # in order to handle both singular and multiple channel output data:
        self._number_of_samples = scanpattern[0].shape[0]


        with ni.Task() as write_task, ni.Task() as read_task, ni.Task() as sample_clk_task:
            # Use a counter output pulse train task as the sample clock source
            # for both the AI and AO tasks.

            # We're stealing the device identifier for the clock from the input mapping string, because of backward
            # compatibility
            sample_clk_task.co_channels.add_co_pulse_chan_freq(
                f"{self._input_mapping.split('/')[0]}/ctr0", freq= self.calculate_sample_rate()
            )
            sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=self._number_of_samples)

            samp_clk_terminal = f"/{self._input_mapping.split('/')[0]}/Ctr0InternalOutput"

            # perhaps remove
            ao_args = {'min_val': self._input_min,
                       'max_val': self._input_max}

            write_task.ao_channels.add_ao_voltage_chan(self._x_mirror_mapping, **ao_args)
            write_task.ao_channels.add_ao_voltage_chan(self._y_mirror_mapping, **ao_args)

            ai_args = {'min_val': self._input_min,
                       'max_val': self._input_max,
                       'terminal_config': ni.constants.TerminalConfiguration.RSE}
            write_task.timing.cfg_samp_clk_timing(
                self.calculate_sample_rate(),
                source=samp_clk_terminal,
                active_edge=Edge.RISING,
                samps_per_chan=self._number_of_samples,
            )

            read_task.ai_channels.add_ai_voltage_chan(self._input_mapping, **ai_args)

            read_task.timing.cfg_samp_clk_timing(
                self.calculate_sample_rate(),
                source=samp_clk_terminal,
                active_edge=Edge.FALLING,
                samps_per_chan=self._number_of_samples,
            )

            writer = AnalogMultiChannelWriter(write_task.out_stream)
            reader = AnalogUnscaledReader(read_task.in_stream)

            writer.write_many_sample(scanpattern)
            # Start the read and write tasks before starting the sample clock
            # source task.

            # IN TRIGGER!
            read_task.start()
            write_task.start()
            sample_clk_task.start()

            # This needs to be separated into some read function
            self.raw_data = np.zeros([1, self._number_of_samples], dtype=np.int16)
            reader.read_int16(
                self.raw_data, number_of_samples_per_channel=self._number_of_samples, timeout=ni.constants.WAIT_INFINITELY)


    def pmt_to_image(self, data):
        """
        Converts 1D PMT signal to a 2D image.

        Parameters:
        data (numpy.ndarray): 1D array of PMT data.

        Returns:
        numpy.ndarray: 2D reconstructed image.
        """


        # Skip the first point if necessary and compensate for delay
        data = data[1:]
        delay = self._delay
        data = np.pad(data, (delay, 0), mode='constant')[:len(data)]

        # Calculate the actual number of steps in x-direction accounting for padding
        x_steps_with_padding = self.data_shape[1] + self._scan_padding
        y_steps = self.data_shape[0]

        # Reshape data to 2D with padding
        image_with_padding = np.reshape(data, (y_steps, x_steps_with_padding))
        image = np.zeros((self.data_shape[0],self.data_shape[1]),dtype='uint16')

        # Remove padding from the image
        if self._scan_padding:
            if self._bidirectional:
                for i in range(y_steps):
                    if i % 2 == 0:
                        image[i] = image_with_padding[i][self._scan_padding:]
                    else:
                        image[i] = np.flip(image_with_padding[i][self._scan_padding:])
            else:
                image = image_with_padding[:, self._scan_padding:]
        else:
            if self._bidirectional:
                for i in range(y_steps):
                    if i % 2 == 0:
                        image[i] = image_with_padding[i]
                    else:
                        image[i] = np.flip(image_with_padding[i])
            else:
                image = image_with_padding

        return image


    def calculate_sample_rate(self):

        # Example calculation
        total_points = self._data_shape[0] * self._data_shape[1]

        return total_points / self._duration.to(u.s).value

    def _do_trigger(self):
        pattern = self.scanpattern()

        self.setup_reader_writer(pattern)

    def _fetch(self, out: Union[np.ndarray, None]) -> np.ndarray:
        # ToDo: Do the actual reading here
        if out is None:
            out = self.pmt_to_image(self.raw_data.flatten())
        else:
            out[...] = self.pmt_to_image(self.raw_data.flatten())
        return out

    def _update_pixel_size(self):
        # Calculate pixel size based on zoom, width, and height
        # check this with tom
        # perhaps make property of the magic number 800
        self._pixel_size = (self._full_scan_range / (self._zoom * self.data_shape[0]))

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
        return self._delay # add unit

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

scanner = LaserScanning(x_mirror_mapping='Dev4/ao0', y_mirror_mapping='Dev4/ao1', input_mapping='Dev4/ai0')
devices = {'cam': scanner}

