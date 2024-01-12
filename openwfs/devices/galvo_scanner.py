from typing import Union, Sequence, Optional
import astropy.units as u
import nidaqmx.system
import numpy as np
from astropy.units import Quantity
from ..core import Detector
from ..slm.patterns import coordinate_range
import nidaqmx as ni
from nidaqmx.constants import TaskMode

from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter


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
    A binning of 100 corresponds to the `data_shape` that is passed in the initializer of the object.

    The image quality at the edge of the scan range is usually low due to the fact
    that the mirror is rapidly changing the scan direction.
    The `padding` attribute corresponds to a fraction of the full voltage range along the fast axis (axis 1).
    This part of the data is discarded after the measurement.
    Note: the padding affects the physical size of the ROI, and hence the pixel_size
    The padding does not affect the number of pixels (`data_shape`) in the returned image.

    Properties:
        left (int): The leftmost pixel of the Region of Interest (ROI) in the scan range.
        top (int): The topmost pixel of the ROI in the scan range.
        height (int): The number of pixels in the vertical dimension of the ROI.
        width (int): The number of pixels in the horizontal dimension of the ROI.
        dwell_time (Quantity[u.us]): The time spent on each pixel during scanning.
        duration (Quantity[u.ms]): Total duration of scanning for one frame.
        delay (Quantity[u.us]): Delay between the control signal to the mirrors and the start of data acquisition.
        binning (int): Factor by which the resolution is reduced; lower binning increases resolution.
        padding (float): Fraction of the scan range at the edges to discard to reduce edge artifacts.
        bidirectional (bool): Whether scanning is bidirectional along the fast axis.
        zoom (float): Used to zoom in at the center of the ROI.

    """

    # LaserScanner(input=("ai/8", -1.0 * u.V, 1.0 * u.V))
    def __init__(self,
                 data_shape: Sequence[int],
                 input: tuple[str, Quantity[u.V], Quantity[u.V]],  # noqa
                 axis0: tuple[str, Quantity[u.V], Quantity[u.V]],
                 axis1: tuple[str, Quantity[u.V], Quantity[u.V]],
                 scale: Quantity[u.um / u.V],
                 sample_rate: Quantity[u.Hz],
                 delay: float = 0.0,
                 padding: float = 0.05,
                 bidirectional: bool = True,
                 simulation: Optional[str] = None):
        """
        Args:
            data_shape (tuple[int, int]): number of data points (height, width) in the full field of view.
                Note that the ROI can be reduced later by setting width, height, top and left,
                and the resolution can be changed by modifying the `binning` property.
            input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the Nidaq channel to use for the input.
                 Tuple of: (name of the channel (e.g., 'ai/1'), minimum voltage, maximum voltage).
            axis0: tuple[str, Quantity[u.V], Quantity[u.V], TerminalConfiguration],
                 Description of the Nidaq channel to use for controlling the axis 0 galvo (slow axis).
                 The TerminalConfiguration element is optional and defaults to TerminalConfiguration.DEFAULT.
            axis1: tuple[str, Quantity[u.V], Quantity[u.V], TerminalConfiguration],
                 Description of the Nidaq channel to use for controlling the axis 1 galvo (fast axis).
                 The TerminalConfiguration element is optional and defaults to TerminalConfiguration.DEFAULT.
            scale (u.um / u.V):
                Conversion factor between voltage at the NiDaq card and displacement of the focus in the object plane.
                This may be an array of (height, width) conversion factors if the factors differ for the different axes.
                This factor is used to convert pixel positions `x_i, y_i` to voltages,
                 using the equation `(pos + [y_i, x_i]) * pixel_size / scale.
            sample_rate (u.Hz):
                Sample rate of the NiDaq input channel.
                The sample rate affects the total time needed to scan a single frame.
                Setting the ROI, padding, or binning does not affect the sample rate.
            delay (float): Delay between mirror control and data acquisition, measured in pixels
            padding (float): Padding fraction at the sides of the scan. The scanner will scan a larger area than will be
                reconstructed, to make sure the reconstructed image is within the linear scan range of the mirrors.
            bidirectional (bool): If true, enables bidirectional scanning along the fast axis.
        """
        # settings for input/output channels
        self._in_channel = input[0]
        self._in_v_min = input[1].to(u.V)
        self._in_v_max = input[2].to(u.V)
        self._in_terminal_configuration = input[3] if len(input) > 3 else TerminalConfiguration.DEFAULT
        self._axis0_channel = axis0[0]
        self._axis1_channel = axis1[0]
        self._out_v_min = Quantity((axis0[1], axis1[1])).to(u.V)
        self._out_v_max = Quantity((axis0[2], axis1[2])).to(u.V)
        self._sample_rate = sample_rate.to(u.Hz)

        self._scale = scale.repeat(2) if scale.size == 1 else scale

        v_width = (self._out_v_max - self._out_v_min)
        self._padding = np.array((0.0, padding))
        self._v_origin = self._out_v_min + v_width * self._padding * 0.5  # Voltage corresponding to (top,left)=(0,0)
        self._roi_start = Quantity((0.0, 0.0), u.V)  # ROI start position relative to origin
        self._roi_end = v_width * (1.0 - self._padding)  # ROI end position relative to origin

        self._resized = True  # indicate that the output signals need to be recomputed
        self._binning = 1
        self._original_data_shape = data_shape

        # Scan settings
        self._delay = float(delay)
        self._padded_data_shape = np.array((0, 0))
        self._zoom = 1.0
        self._bidirectional = bidirectional

        self._write_task = None
        self._read_task = None

        self._valid = False  # indicates that `trigger()` should initialize the nidaq tasks and scan pattern
        self._scan_pattern = None
        self._simulation = simulation

        self._original_pixel_size = ((self._out_v_max - self._out_v_min) * self._scale / data_shape).to(u.um)
        super().__init__(data_shape=data_shape, pixel_size=self._original_pixel_size, duration=0.0 * u.ms)
        self._update()

    def _update(self):
        # round padding up to integer number of pixels on both sides
        padding = 2 * np.ceil(self._data_shape * self._padding * 0.5).astype('int32')
        self._padded_data_shape = self._data_shape + padding

        # compute the scan range.
        scan_range = (self._roi_end - self._roi_start) * (1.0 + padding / self._data_shape)  # noqa
        center = 0.5 * (self._roi_start + self._roi_end) + self._v_origin
        (voltages0, voltages1a) = coordinate_range(self._padded_data_shape, scan_range, offset=center)

        # clip to voltage limits to prevent possible damage
        voltages0 = voltages0.ravel().clip(self._out_v_min[0], self._out_v_max[0])
        voltages1a = voltages1a.ravel().clip(self._out_v_min[0], self._out_v_max[0])

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
        self._duration = (sample_count / self._sample_rate).to(u.ms)

        if self._simulation is not None:
            return
        # Sets up Nidaq task and i/o channels
        if self._read_task:
            self._read_task.close()
            self._read_task = None
        if self._write_task:
            self._write_task.close()
            self._write_task = None

        self._write_task = ni.Task()
        self._read_task = ni.Task()
        self._read_task.in_stream.timeout = self.timeout

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
                                                        terminal_config=self._in_terminal_configuration)
        self._read_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)
        self._read_task.triggers.start_trigger.cfg_dig_edge_start_trig(self._write_task.triggers.start_trigger.term)
        delay = (self._delay / self._sample_rate).to_value(u.s)
        if delay > 0.0:
            self._read_task.triggers.start_trigger.delay = delay
            self._read_task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS

        self._writer = AnalogMultiChannelWriter(self._write_task.out_stream)
        self._valid = True

    def _do_trigger(self):
        if not self._valid:
            self._update()

        if self._simulation is not None:
            return

        self._read_task.wait_until_done()
        self._write_task.wait_until_done()

        # write the samples to output in the x-y channels
        print(self._scan_pattern.size)
        self._writer.write_many_sample(self._scan_pattern)

        # Start the tasks
        self._read_task.start()  # waits for trigger coming from the write task
        self._write_task.start()

    def _raw_to_cropped(self, raw):
        """Converts the raw scanner data back into a 2-Dimensional image.

        Because the scanner can return both signed and unsigned integers, both cases are accounted for.
        This function crops the data if padding was added, and it
        flips the even rows back if scanned in bidirectional mode.
        """
        start = (self._padded_data_shape - self._data_shape) // 2
        end = self._padded_data_shape - start

        # Change the data type into uint16 if necessary
        if type(raw[0]) == np.int16:
            # add 32768 to go from -32768-32767 to 0-65535
            cropped = raw.reshape(self._padded_data_shape).view('uint16')[start[0]:end[0], start[1]:end[1]] + 0x8000
        elif type(raw[0]) == np.uint16:
            cropped = raw.reshape(self._padded_data_shape)[start[0]:end[0], start[1]:end[1]]
        else:
            raise ValueError('Only int16 and uint16 data types are supported at the moment.')

        if self._bidirectional:
            cropped[1::2, :] = cropped[1::2, ::-1]

        return cropped

    def _fetch(self, out: np.ndarray | None) -> np.ndarray:  # noqa
        """Reads the acquired data from the input task."""
        if self._simulation is None:
            raw = self._read_task.in_stream.read()
            self._read_task.stop()
            self._write_task.stop()
        elif self._simulation == 'horizontal':
            raw = (self._scan_pattern[1, :] * 1000.0).round().astype('int16')  # in mV
        elif self._simulation == 'vertical':
            raw = (self._scan_pattern[0, :] * 1000.0).round().astype('int16')
        else:
            raise ValueError(
                f"Invalid simulation option {self._simulation}. Should be 'horizontal', 'vertical', or 'None'")

        cropped = self._raw_to_cropped(raw)

        if out is not None:
            out[...] = cropped
        else:
            out = cropped

        return out

    @property
    def left(self) -> int:
        """The leftmost pixel of the Region of Interest (ROI) in the scan range."""
        return int(np.round(self._roi_start[1] * self._scale[1] / self._pixel_size[1]))

    @left.setter
    def left(self, value: int):
        self._roi_start[1] = value * self._pixel_size[1] / self._scale[1]
        self._valid = False

    @property
    def top(self) -> int:
        """The topmost pixel of the ROI in the scan range."""
        return int(np.round(self._roi_start[0] * self._scale[0] / self._pixel_size[0]))

    @top.setter
    def top(self, value: int):
        self._roi_start[0] = value * self._pixel_size[0] / self._scale[0]
        self._valid = False

    @property
    def height(self) -> int:
        """The number of pixels in the vertical dimension of the ROI."""
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._data_shape = (int(value), int(self.data_shape[1]))
        self._roi_end = self._roi_start + self._pixel_size * self._data_shape / self._scale
        self._valid = False

    @property
    def width(self) -> int:
        """The number of pixels in the horizontal dimension of the ROI."""
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._data_shape = (self.data_shape[0], int(value))
        self._roi_end = self._roi_start + self._pixel_size * self._data_shape / self._scale
        self._valid = False

    @property
    def dwell_time(self) -> Quantity[u.us]:
        """The time spent on each pixel during scanning."""
        return (1.0 / self._sample_rate).to(u.us)

    @dwell_time.setter
    def dwell_time(self, value: Quantity[u.us]):
        old = self._sample_rate
        self._sample_rate = (1.0 / value).to(u.Hz)
        self._delay = float(self._delay / old * self._sample_rate)
        self._update()  # to update duration

    @property
    def delay(self) -> float:
        """Delay between the control signal to the mirrors and the start of data acquisition."""
        return self._delay  # add unit

    @delay.setter
    def delay(self, value: float):
        self._delay = value
        self._valid = False

    @property
    def padding(self) -> float:
        """Fraction of the scan range at the edges to discard to reduce edge artifacts."""
        return self._padding[1]

    @padding.setter
    def padding(self, value: float):
        self._padding[1] = value
        self._v_origin = self._out_v_min + (self._out_v_max - self._out_v_min) * self._padding * 0.5
        self._valid = False

    @property
    def bidirectional(self) -> bool:
        """Whether scanning is bidirectional along the fast axis."""
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value: bool):
        self._bidirectional = value
        self._valid = False

    @property
    def zoom(self) -> float:
        """Zoom factor.
        The zoom factor determines the pixel size relative to the original pixel size.
        The original pixel size is given by `_scale * (_out_v_max - _out_v_min) / _data_shape`
        When the zoom factor is changed,
        the center of the region of interest and the number of pixels in the data remain constant.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        roi_center = 0.5 * (self._roi_end + self._roi_start)
        roi_width_old = self._roi_end - self._roi_start
        roi_width_new = roi_width_old * self._zoom / value
        self._roi_start = roi_center - 0.5 * roi_width_new
        self._roi_end = roi_center + 0.5 * roi_width_new
        self._zoom = float(value)
        self._pixel_size = self._original_pixel_size * self._binning / self._zoom
        self._valid = False

    @property
    def binning(self) -> int:
        """Undersampling factor.

        Increasing the binning reduces the number of pixels in the image while keeping dwell time the same.
        As a result, the total duration of a scan decreases.
        Note: this behavior is different from that of a real camera.
            No actual binning is performed, the scanner just takes fewer steps in x and y

        Note: the ROI is kept the same as much as possible.
            However, due to rounding, it may vary slightly.

        """
        return self._binning

    @binning.setter
    def binning(self, value: int):
        ratio = self._binning / value
        self._binning = value
        self._data_shape = tuple(int(s) for s in np.round(np.array(self._data_shape) * ratio))
        self._pixel_size = self._original_pixel_size * self._binning / self._zoom
        self._roi_end = self._roi_start + self._pixel_size * self._data_shape / self._scale
        self._valid = False

    @staticmethod
    def list_devices():
        return [d.name for d in nidaqmx.system.System().devices]
