from typing import Union, Sequence
import astropy.units as u
import nidaqmx.system
import numpy as np
from astropy.units import Quantity
from ..core import Detector, unitless
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


class ScanningMicroscope(Detector):
    """Controller for a laser-scanning microscope with two galvo mirrors
    controlled by a National Instruments data acquisition card (nidaq).

    Effectively, a `ScanningMicroscope` works like a camera, which can be triggered and returns 2-D images.
    These images are obtained by raster-scanning a focus using two galvo mirrors, controlled by a nidaq card,
    and recording a detector signal (typically from a photon multiplier tube (PMT)) with the same card.

    Upon construction, the maximum scan range (in volt) is set for both axes,
    and a scale parameter should be specified to convert these voltages to positions in the object plane.
    The number of pixels in the full scan range is specified with the `data_shape` parameter.

    After construction, a sub-region of the scan range (i.e., a region of interest ROI) may be selected using
    the `top`, `left`, `height` and `width` properties.
    Note that changing the ROI does not change the pixel size.
    Only the number of pixels in the returned image changes, as with a regular camera.

    The `binning` property can be used to change the pixel size without changing the ROI.
    By decreasing the `binning` property, more points are measured for the same ROI,
    thereby increasing the resolution.
    For compatibility with micromanager, the `binning` property should be an integer.
    A binning of 100 corresponds to the `data_shape` passed in the initializer of the object.

    The image quality at the edge of the scan range is usually low due to the fact
    that the mirror is rapidly changing the scan direction.
    The `padding` attribute corresponds to a fraction of the full voltage range along the fast axis (axis 1).
    This part of the data is discarded after the measurement.
    Note: the padding affects the physical size of the ROI, and hence the pixel_size
    The padding does not affect the number of pixels (`data_shape`) in the returned image.

    Delay
    """

    # LaserScanner(input=("ai/8", -1.0 * u.V, 1.0 * u.V))
    def __init__(self,
                 data_shape: Sequence[int],
                 input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 axis0: tuple[str, Quantity[u.V], Quantity[u.V]],
                 axis1: tuple[str, Quantity[u.V], Quantity[u.V]],
                 scale: Quantity[u.um / u.V],
                 sample_rate: Quantity[u.Hz],
                 delay=0.0 * u.us,
                 binning=64,
                 padding=0.05, bidirectional=True):
        """
        Args:
            data_shape (tuple[int,int]): number of data points (height, width) of the full field of view.
                Note that the ROI can be reduced later by setting width, height, top and left,
                and the resolution can be changed by modifying the `binning` property.
            input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for the input.
                 Tuple of: (name of the channel (e.g., 'ai/1'), minimum voltage, maximum voltage).
            axis0: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for controlling the axis 0 galvo (slow axis).
            axis1: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for controlling the axis 1 galvo (fast axis).
            scale (u.um / u.V):
                Conversion factor between voltage at the NiDaq card and displacement of the focus in the object plane.
                This may be an array of (height, width) conversion factors if the factors differ for the different axes.
                This factor is used to convert pixel positions `x_i, y_i` to voltages,
                 using the equation `(pos + [y_i, x_i]) * pixel_size / scale.
            sample_rate (u.Hz):
                Sample rate of the NiDaq input channel.
                The sample rate affects the total time needed to scan a single frame.
                Setting the ROI, padding, or binning does not affect the sample rate.
            delay:
            padding:
            bidirectional:
        """
        # settings for input/output channels
        self._in_channel = input[0]
        self._in_v_min = input[1].to(u.V)
        self._in_v_max = input[2].to(u.V)
        self._axis0_channel = axis0[0]
        self._axis1_channel = axis1[0]
        self._out_v_min = Quantity((axis0[1], axis1[1])).to(u.V)
        self._out_v_max = Quantity((axis0[2], axis1[2])).to(u.V)
        self._sample_rate = sample_rate.to(u.Hz)

        self._scale = scale.repeat(2) if scale.size == 1 else scale
        self._full_resolution = np.array(data_shape) * binning

        v_width = (self._out_v_max - self._out_v_min)
        self._padding = np.array((0.0, padding))
        self._v_origin = self._out_v_min + v_width * self._padding * 0.5  # Voltage corresponding to (top,left)=(0,0)
        self._roi_start = Quantity((0.0, 0.0), u.V)  # ROI start position relative to origin
        self._roi_end = v_width * (1.0 - self._padding)  # ROI end position relative to origin

        self._resized = True  # indicate that the output signals need to be recomputed
        self._binning = 100

        # Scan settings
        self._delay = delay.to(u.us)
        self._padded_data_shape = (0, 0)
        self._bidirectional = bidirectional

        self._write_task = None
        self._read_task = None

        self._valid = False  # indicates that `trigger()` should initialize the nidaq tasks and scan pattern
        self._scan_pattern = None

        super().__init__(data_shape=data_shape, pixel_size=np.array((0.0, 0.0)) * u.um, duration=0.0 * u.ms)
        self._update()

    def _update(self):
        v_extent = self._out_v_max - self._out_v_min

        # round padding up to integer number of pixels on both sides
        padding = 2 * np.ceil(self._data_shape * self._padding * 0.5).astype('int32')
        self._padded_data_shape = self._data_shape + padding

        # compute the scan range.
        # first compute the padding voltage corresponding to the rounded padding (_padded_data_shape)
        # then construct a voltage range from _v_origin + _roi_start - 0.5 * v_padding
        # to _v_origin + _roi_end + 0.5 * v_padding
        #
        v_padding = v_extent * padding / self._data_shape
        scan_start = self._v_origin + self._roi_start - 0.5 * v_padding
        scan_range = self._roi_end - self._roi_start + v_padding
        voltages0 = scan_start[0] + scan_range[0] * ((np.arange(self._padded_data_shape[0]) + 0.5) /
                                                     self._padded_data_shape[0])

        voltages1a = scan_start[1] + scan_range[1] * ((np.arange(self._padded_data_shape[1]) + 0.5) /
                                                      self._padded_data_shape[1])

        # crop to voltage limits to prevent possible damage
        voltages0 = np.clip(voltages0, self._out_v_min[0], self._out_v_max[0])
        voltages1a = np.clip(voltages1a, self._out_v_min[0], self._out_v_max[0])

        # for bidirectional scanning, reverse the direction of the even scan lines
        if self._bidirectional:
            voltages1b = voltages1a[::-1]
        else:
            voltages1b = voltages1a

        # Generate output voltages
        self._scan_pattern = np.zeros((2, *self._padded_data_shape))
        self._scan_pattern[0, :, :] = voltages0.to_value(u.V).reshape((-1, 1))
        self._scan_pattern[1, 0::2, :] = voltages1a.to_value(u.V).reshape((1, -1))
        self._scan_pattern[1, 1::2, :] = voltages1b.to_value(u.V).reshape((1, -1))
        self._scan_pattern = self._scan_pattern.reshape(2, -1)

        sample_count = self._padded_data_shape[0] * self._padded_data_shape[1]
        self._pixel_size = ((self._roi_end - self._roi_start) * self._scale / self._data_shape).to(u.um)
        self._duration = (sample_count / self._sample_rate).to(u.ms)

        # Sets up Nidaq task and i/o channels
        if self._read_task:
            self._read_task.close()
            self._read_task = None
        if self._write_task:
            self._write_task.close()
            self._write_task = None

        self._write_task = ni.Task()
        self._read_task = ni.Task()

        # Configure the sample clock task
        sample_rate = self._sample_rate.to_value(u.Hz)

        # Configure the analog output task (two channels)
        self._write_task.ao_channels.add_ao_voltage_chan(self._axis0_channel,
                                                         min_val=self._out_v_min[0].to_value(u.V),
                                                         max_val=self._out_v_max[0].to_value(u.V))
        self._write_task.ao_channels.add_ao_voltage_chan(self._axis1_channel,
                                                         min_val=self._out_v_min[1].to_value(u.V),
                                                         max_val=self._out_v_max[1].to_value(u.V))
        self._write_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)

        # Configure the analog input task (one channel)
        self._read_task.ai_channels.add_ai_voltage_chan(self._in_channel,
                                                        min_val=self._in_v_min.to_value(u.V),
                                                        max_val=self._in_v_max.to_value(u.V),
                                                        terminal_config=TerminalConfiguration.RSE)
        self._read_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)
        self._read_task.triggers.start_trigger.cfg_dig_edge_start_trig(self._write_task.triggers.start_trigger.term)
        delay = self._delay.to_value(u.s)
        if delay > 0.0:
            self._read_task.triggers.start_trigger.delay = delay
            self._read_task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS

        self._writer = AnalogMultiChannelWriter(self._write_task.out_stream)
        self._reader = AnalogUnscaledReader(self._read_task.in_stream)

    def _do_trigger(self):
        if not self._valid:
            self._update()
            self._valid = True

        self._read_task.wait_until_done()
        self._write_task.wait_until_done()

        # write the samples to output in the x-y channels
        print(self._scan_pattern.size)
        self._writer.write_many_sample(self._scan_pattern)

        # Start the tasks
        self._read_task.start()  # waits for trigger coming from the write task
        self._write_task.start()

    def _fetch(self, out: Union[np.ndarray, None]) -> np.ndarray:  # noqa
        """Reads the acquired data from the input task."""
        sample_count = self._padded_data_shape[0] * self._padded_data_shape[1]
        raw = np.zeros((1, sample_count), dtype=np.int16)
        self._reader.read_int16(raw, number_of_samples_per_channel=sample_count,
                                timeout=ni.constants.WAIT_INFINITELY)
        self._read_task.stop()
        self._write_task.stop()
        start = (self._padded_data_shape - self._data_shape) // 2
        end = self._padded_data_shape - start
        cropped = raw.reshape(self._padded_data_shape).view(dtype='uint16')[start[0]:end[0], start[1]:end[1]] + 0x7FFF
        if self._bidirectional:
            cropped[1::2, :] = cropped[1::2, ::-1]

        if out is not None:
            out[...] = cropped
        else:
            out = cropped

        return out

    @property
    def left(self) -> int:
        return np.round(self._roi_start[1] * self._scale[1] / self._pixel_size[1]).astype('int32')

    @left.setter
    def left(self, value: int):
        self._roi_start[1] = value * self._pixel_size[1] / self._scale[1]
        self._valid = False

    @property
    def top(self) -> int:
        return np.round(self._roi_start[0] * self._scale[0] / self._pixel_size[0]).astype('int32')

    @top.setter
    def top(self, value: int):
        self._roi_start[0] = value * self._pixel_size[0] / self._scale[0]
        self._valid = False

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._data_shape = (value, self.data_shape[1])
        self._roi_end = self._roi_start + self._pixel_size * self._data_shape / self._scale
        self._valid = False

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._data_shape = (self.data_shape[0], value)
        self._roi_end = self._roi_start + self._pixel_size * self._data_shape / self._scale
        self._valid = False

    @property
    def dwell_time(self) -> Quantity[u.us]:
        return (1.0 / self._sample_rate).to(u.us)

    @dwell_time.setter
    def dwell_time(self, value: Quantity[u.us]):
        self._sample_rate = (1.0 / value).to(u.Hz)
        self._valid = False

    @property
    def duration(self) -> Quantity[u.ms]:
        return self._duration.to(u.ms)

    @duration.setter
    def duration(self, value):
        self._duration = value.to(u.ms)

    @property
    def delay(self) -> Quantity[u.us]:
        return self._delay  # add unit

    @delay.setter
    def delay(self, value: Quantity[u.us]):
        self._delay = value
        self._valid = False

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: float):
        self._binning = value
        self._data_shape = tuple(np.round(self._full_resolution / self._binning).astype('int32'))
        self._update()  # to recompute changed pixel size directly

    @property
    def padding(self) -> float:
        return self._padding[1]

    @padding.setter
    def padding(self, value: float):
        self._padding[1] = value
        self._v_origin = self._out_v_min + (self._out_v_max - self._out_v_min) * self._padding * 0.5
        self._valid = False

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value: bool):
        self._bidirectional = value
        self._valid = False

    @staticmethod
    def list_devices():
        return [d.name for d in nidaqmx.system.System().devices]
