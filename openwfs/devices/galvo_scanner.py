from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import nidaqmx as ni
import nidaqmx.system
import numpy as np
from astropy.units import Quantity
from nidaqmx.constants import TaskMode
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from numpy._typing import ArrayLike

from ..core import Detector
from ..utilities import unitless


@dataclass
class Axis:
    channel: str
    v_min: Quantity[u.V]
    v_max: Quantity[u.V]
    maximum_acceleration: Quantity[u.V / u.s ** 2]
    terminal_configuration: TerminalConfiguration = TerminalConfiguration.DEFAULT

    def to_volt(self, pos: ArrayLike) -> Quantity[u.V]:
        """Converts relative position [0.0 .. 1.0] to voltage [V_min .. V_max]

        Currently, this is just a linear conversion, but a lookup table may be used in the future.
        """
        return self.v_min + np.clip(pos, 0.0, 1.0) * (self.v_max - self.v_min)

    def _to_pos(self, volt: Quantity[u.V]) -> ArrayLike:
        """Converts voltage [V_min .. V_max] to relative position [0.0 .. 1.0]"""
        return (volt - self.v_min) / (self.v_max - self.v_min)

    def step(self, start: float, stop: float, sample_rate: Quantity[u.Hz]) -> Quantity[u.V]:
        """
        Generate a voltage sequence to move from `start` to `stop` in
        the fastest way possible.

        This function assumes that the mirror is standing still at v_start,
        and generates a voltage ramp to move the mirror and stop it exactly at v_end,
        using the maximum acceleration allowed by the mirror.

        The voltage curve is given by:
        v_start + 1/2 a·t²              for t < t_total /2
        v_end - 1/2 a·(t_total-t)²  for t >= t_total

        using continuity at t=t_total/2, we can solve this equation to get:
        t_total = 2·sqrt((v_end - v_start) / a)

        Returns:
            Quantity[u.V]: voltage sequence
        """
        v_start = self.to_volt(start)
        v_end = self.to_volt(stop)

        # t is measured in samples
        # a is measured in volt/sample²
        a = self.maximum_acceleration / sample_rate ** 2 * np.sign(v_end - v_start)
        t_total = unitless(2.0 * np.sqrt((v_end - v_start) / a))
        t = np.arange(np.ceil(t_total + 1E-6))  # add a small number to deal with case t=0 (start=end)
        v_accel = v_start + 0.5 * a * t[:len(t) // 2] ** 2  # acceleration part
        v_decel = v_end - 0.5 * a * (t_total - t[len(t) // 2:]) ** 2  # deceleration part
        v_decel[-1] = v_end  # fix last point because t may be > t_total due to rounding
        return np.clip(np.concatenate((v_accel, v_decel)), self.v_min, self.v_max)

    def scan(self, start: float, stop: float, sample_count: int, sample_rate: Quantity[u.Hz]):
        """
        Generate a voltage sequence to scan with a constant velocity from start to stop, including acceleration and deceleration.

        Before starting this sequence, the mirror is assumed to be standing still at the launch point,
        which is some distance _before_ start.
        After the scan sequence, the mirror is stopped at the landing point,
        which is some distance _after_ stop.
        The launch point and landing point are returned along with the scan sequence.

        This function also returns a slice object, which represents the part of the sequence
        that corresponds to a linear movement from start to stop. `slice.stop - slice.start = sample_count`.

        The scan follows the coordinate convention used throughout OpenWFS and Astropy,
        where the coordinates correspond to the centers of the pixels.
        Therefore, while the linear part of the scan starts at start and ends at stop,
        the sample points in this range correspond to the centers of the pixels,
        so the sample at slice.start lies half a pixel _after_ start,
        and the sample at slice.stop - 1 lies half a pixel _before_ stop.

        Returns:
            (Quantity[u.V], float, float, slice): voltage sequence, launch point, landing point, slice object
        """
        v_start = self.to_volt(start)
        v_end = self.to_volt(stop)

        scan_speed = (v_end - v_start) / sample_count  # V per sample
        a = self.maximum_acceleration / sample_rate ** 2 * np.sign(scan_speed)  # V per sample²

        # construct a sequence to accelerate from speed 0 to the scan speed
        # we start by constructing a sequence with a maximum acceleration.
        # This sequence may be up to 1  sample longer than needed to reach the scan speed.
        # This last sample is replaced by movement at a linear scan speed
        t_launch = np.arange(np.ceil(unitless(scan_speed / a)))  # in samples
        v_accel = 0.5 * a * t_launch ** 2  # last 1 to 2 samples may have faster scan speed than needed
        if np.abs(v_accel[-1] - v_accel[-2]) > np.abs(scan_speed):
            v_accel[-1] = v_accel[-2] + scan_speed
        v_launch = v_start - v_accel[-1] - 0.5 * scan_speed  # launch point
        v_land = v_end + v_accel[-1] + 0.5 * scan_speed  # landing point

        # linear part of the scan
        v_linear = v_start + scan_speed * (np.arange(sample_count) + 0.5)

        # combine the parts
        v = np.concatenate((v_launch + v_accel, v_linear, v_land - v_accel[::-1]))
        v = np.clip(v, self.v_min, self.v_max)
        launch = self._to_pos(v_launch)
        land = self._to_pos(v_land)
        return v, launch, land, slice(len(v_accel), len(v_accel) + sample_count)


class ScanningMicroscope(Detector):
    """Controller for a laser-scanning microscope with two galvo mirrors
    controlled by a National Instruments data acquisition card (nidaq).

    Effectively, a `ScanningMicroscope` works like a camera, which can be triggered and returns 2-D images.
    These images are obtained by raster-scanning a focus using two galvo mirrors, controlled by a nidaq card,
    and recording a detector signal (typically from a photon multiplier tube (PMT)) with the same card.

    Upon construction, the maximum voltage range is set for both axes.
    To avoid possible damage to the hardware, this range cannot be exceeded during scanning,
    and the value cannot be changed after construction of the ScanningMicroscope.

    It is recommended to set this voltage range slightly _larger_ than the maximum field of view
    of the microscope (but still within the limits that may cause damage to the hardware),
    and then set the `reference_zoom` so that the exact field of view is covered.
    The user can later choose a smaller field of view in two ways:
    1. by setting the zoom factor. This will keep the number of pixels the same, and thus change the pixel size
    2. by changing the roi width and height. This will change the number of pixels, but keeps the pixel size the same.

    Example:
        >>> # construct the microscope object.
        >>> # The voltage range is larger than the field of view, but small enough not to damage anything
        >>> # The reference zoom factor is set to the amount we need to zoom in to exactly scan the maximum field of view.
        >>> # So, assuming that the full voltage range corresponds to overscanning the field of view by 10%, we set a reference_zoom of 1.1.
        >>> # The zoom property of the microscope is multiplied by the reference zoom factor, so that zoom=1.0 corresponds to the full field of view
        >>> microscope = ScanningMicroscope(
        >>>         axis0=('Dev4/ao0', -2.0 * u.V, 2.0 * u.V),
        >>>         axis1=('Dev4/ao1', -2.0 * u.V, 2.0 * u.V),
        >>>         input=('Dev4/ai0', 0.1 * u.V, 1.0 * u.V),
        >>>         sample_rate=0.5 * u.MHz,
        >>>         data_shape=(500,500),
        >>>         reference_zoom=1.1,
        >>>         scale=440 * u.um / u.V)
        >>>
        >>> # crop the region of interest to the maximum field of view
        >>> full_fov = (50, 50, 400, 400)
        >>> microscope.roi = full_fov
        >>>
        >>> # add some overscan to reduce edge artifacts caused by the mirror changing direction
        >>> # this will cause the mirror to move slightly beyond the region of interest in the top-left corner
        >>> # and the data will be cropped to the region of interest after the scan
        >>> # this way, the mirror is moving at a constant speed when scanning the region of interest.
        >>> microscope.overscan_x = 30  # fast axis, overscan in pixels
        >>> microscope.overscan_y = 1  # slow axis, overscan in pixels
        >>>
        >>> # trigger the microscope and read the image
        >>> frame = microscope.read()
        >>>
        >>> # select a specific region of interest
        >>> microscope.roi = (100, 100, 200, 200)
        >>>
        >>> # Switch to a higher resolution (half the pixel size).
        >>> # Note: this does not affect the physical size or position of roi, but it does double the number of pixels in the roi
        >>> # therefore, the roi is now at (200, 200, 400, 400)
        >>> microscope.binning = microscope.binning // 2

    The image quality at the start of the scan range is usually low due to the fact
    that the mirror requires some time to return to the start position (for the slow axis or unidirectional scans)
    or requires some time to change scan direction (for the fast axis of bidirectional scans)
    To reduce these artifacts, the overscan_x and overscan_y properties can be set.
    These properties specify the number
    has justed changed scan direction and mayis rapidly changing the scan direction.
    The `overscanpadding` attribute corresponds to a fraction of the full voltage range along the fast axis (axis 1).
    This part of the data is discarded after the measurement.
    Note: the padding affects the physical size of the ROI, and hence the pixel_size
    The padding does not affect the number of pixels (`data_shape`) in the returned image.

    Example:
        >>> scanner = ScanningMicroscope(
                                bidirectional=True,
                                sample_rate=0.5 * u.MHz,
                                axis0=('Dev4/ao0', -2.0 * u.V, 2.0 * u.V),
                                axis1=('Dev4/ao1', -2.0 * u.V, 2.0 * u.V),
                                input=('Dev4/ai0', -1.0 * u.V, 1.0 * u.V),
                                data_shape=(1000, 1000),
                                scale=440 * u.um / u.V)
        >>> frame = scanner.read()
    """

    # LaserScanner(input=("ai/8", -1.0 * u.V, 1.0 * u.V))
    def __init__(self,
                 input: tuple[str, Quantity[u.V], Quantity[u.V]],  # noqa
                 axis0: Axis,
                 axis1: Axis,
                 scale: Quantity[u.um / u.V],
                 sample_rate: Quantity[u.MHz],
                 base_resolution: int = 512,
                 delay: Quantity[u.us] = 0.0 * u.us,
                 reference_zoom: float = 1.0,
                 bidirectional: bool = True,
                 simulation: Optional[str] = None,
                 multi_threaded: bool = True,
                 preprocess: Optional[callable] = None):
        """
        Args:
            base_resolution (int): number of pixels (height and width) in the full field of view.
                Note that the ROI can be reduced later by setting width, height, top and left.
            input: tuple[str, Quantity[u.V], Quantity[u.V]],
                 Description of the NI-DAQ channel to use for the input.
                 Tuple of: (name of the channel (e. g., 'ai/1'), minimum voltage, maximum voltage).
            axis0: Axis
                 Description of the NI-DAQ channel to use for controlling the slow axis.
            axis1: Axis
                 Description of the NI-DAQ channel to use for controlling the fast axis.
            scale (u.um / u.V):
                Conversion factor between voltage at the NI-DAQ card and displacement of the focus in the object plane.
                This may be an array of (height, width) conversion factors if the factors differ for the different axes.
            sample_rate (u.Hz):
                Sample rate of the NI-DAQ input channel.
            delay (u.us): Delay between mirror control and data acquisition, measured in microseconds
            reference_zoom (float): Zoom factor that corresponds to fitting the full field of view exactly.
                The zoom factor in the `zoom` property is multiplied by the `reference_zoom` to compute the scan range.
            bidirectional (bool): If true, enables bidirectional scanning along the fast axis.
            preprocess (callable): Process the raw data with this function before cropping. When None, the preprocessing
                will be skipped. The function must take input arguments data and sample_rate, and must return the
                preprocessed data.
        """
        self._y_axis = axis0
        self._x_axis = axis1
        (self._input_channel, self._input_v_min, self._input_v_max) = input
        self._input_terminal_configuration = input[3] if len(input) == 4 else TerminalConfiguration.DEFAULT
        self._scale = scale.repeat(2) if scale.size == 1 else scale
        self._sample_rate = sample_rate.to(u.MHz)
        self._binning = 1  # binning factor = sample_rate · dwell_time
        self._base_resolution = int(base_resolution)
        self._roi_top = 0
        self._roi_left = 0
        self._delay = delay.to(u.us)
        self._reference_zoom = float(reference_zoom)
        self._zoom = 1.0
        self._bidirectional = bidirectional
        self._simulation = simulation
        self._preprocess = preprocess

        self._write_task = None
        self._read_task = None

        self._valid = False  # indicates that `trigger()` should initialize the NI-DAQ tasks and scan pattern
        self._scan_pattern = None

        # the pixel size and duration are computed dynamically
        # data_shape just returns self._data shape, and latency = 0.0 ms
        super().__init__(data_shape=(base_resolution, base_resolution), pixel_size=None, duration=None,
                         latency=0.0 * u.ms,
                         multi_threaded=multi_threaded)
        self._update()

    def _update(self):
        """Computes the scan pattern"""

        width = self._data_shape[1]
        height = self._data_shape[0]
        roi_scale = 1.0 / (self._reference_zoom * self._zoom) / self._base_resolution
        roi_left = self._roi_left * roi_scale
        roi_right = (self._roi_left + width) * roi_scale
        roi_top = self._roi_top * roi_scale
        roi_bottom = (self._roi_top + height) * roi_scale

        # Compute the retrace pattern for the slow axis
        v_yr = self._y_axis.step(roi_bottom, roi_top, self._sample_rate)

        # Compute the scan pattern for the fast axis
        v_x_even, x_launch, x_land, self._mask = self._x_axis.scan(roi_left, roi_right, width, self._sample_rate)
        if self._bidirectional:
            v_x_odd, _, _, _ = self._x_axis.scan(roi_right, roi_left, width, self._sample_rate)
        else:
            v_xr = self._x_axis.step(x_land, x_launch, self._sample_rate)  # horizontal retrace
            v_x_even = np.concatenate((v_x_even, v_xr))
            v_x_odd = v_x_even

        # Set voltages for the scan.
        # The horizontal scan pattern consists of alternating even/odd scan lines
        # For unidirectional mode, these are the same
        # For bidirectional mode, the scan pattern is padded to always have an even number of scan lines
        # The horizontal pattern is repeated continuously, so even during the
        # vertical retrace. In bidirectional scan mode, th
        n_rows = self._data_shape[0] + np.ceil(len(v_yr) / len(v_x_odd)).astype('int32')
        self._n_cols = len(v_x_odd)
        if self._bidirectional and n_rows % 2 == 1:
            n_rows += 1

        scan_pattern = np.zeros((2, n_rows, self._n_cols))
        scan_pattern[1, 0::2, :] = v_x_even  # .reshape((1, -1))
        scan_pattern[1, 1::2, :] = v_x_odd

        # todo: use function from patterns / utilities
        y_coord = (np.arange(height) + 0.5) * roi_scale + roi_top
        scan_pattern[0, :height, :] = self._y_axis.to_volt(y_coord).reshape(-1, 1)

        # The last row(s) are used for the vertical retrace
        # We park the mirror after the vertical retrace, to allow
        # the horizontal scan mirror to finish its scan.
        # Note: this may not always be needed, but it guarantees
        # that the horizontal scan mirror is always scanning at the same frequency
        # which is essential for resonant scanning.
        retrace = scan_pattern[0, height:, :].reshape(-1)
        retrace[0:len(v_yr)] = v_yr
        retrace[len(v_yr):] = v_yr[-1]
        self._scan_pattern = scan_pattern.reshape(2, -1)

        if self._simulation is not None:
            return
        # Sets up NI-DAQ task and i/o channels
        if self._read_task:
            self._read_task.close()
            self._read_task = None
        if self._write_task:
            self._write_task.close()
            self._write_task = None

        self._write_task = ni.Task()
        self._read_task = ni.Task()
        self._read_task.in_stream.timeout = self.timeout.to_value(u.s)

        # Configure the sample clock task
        sample_rate = self._sample_rate.to_value(u.Hz)
        sample_count = self._scan_pattern.shape[1]

        # Configure the analog output task (two channels)
        self._write_task.ao_channels.add_ao_voltage_chan(self._x_axis.channel,
                                                         min_val=self._x_axis.v_min.to_value(u.V),
                                                         max_val=self._x_axis.v_max.to_value(u.V))
        self._write_task.ao_channels.add_ao_voltage_chan(self._y_axis.channel,
                                                         min_val=self._y_axis.v_min.to_value(u.V),
                                                         max_val=self._y_axis.v_max.to_value(u.V))
        self._write_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)

        # Configure the analog input task (one channel)
        self._read_task.ai_channels.add_ai_voltage_chan(self._input_channel,
                                                        min_val=self._input_v_min.to_value(u.V),
                                                        max_val=self._input_v_max.to_value(u.V),
                                                        terminal_config=self._input_terminal_configuration)
        self._read_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)
        self._read_task.triggers.start_trigger.cfg_dig_edge_start_trig(self._write_task.triggers.start_trigger.term)
        delay = self._delay.to_value(u.s)
        if delay > 0.0:
            self._read_task.triggers.start_trigger.delay = delay
            self._read_task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS

        self._writer = AnalogMultiChannelWriter(self._write_task.out_stream)
        self._valid = True

    def _ensure_valid(self):
        if not self._valid:
            self._update()

    def _do_trigger(self):
        """Makes sure scan patterns are up-to-date, and triggers the NI-DAQ tasks."""
        self._ensure_valid()

        if self._simulation is not None:
            return

        self._read_task.wait_until_done()
        self._write_task.wait_until_done()

        # write the samples to output in the x-y channels
        self._writer.write_many_sample(self._scan_pattern)

        # Start the tasks
        self._read_task.start()  # waits for trigger coming from the write task
        self._write_task.start()

    def _raw_to_cropped(self, raw: np.ndarray) -> np.ndarray:
        """Converts the raw scanner data back into a 2-dimensional image.

        Because the scanner can return both signed and unsigned integers, both cases are accounted for.
        This function crops the data if padding was added, and it
        flips the even rows back if scanned in bidirectional mode.
        """
        # convert data to 2-d, discard padding
        cropped = raw.reshape(-1, self._n_cols)[:self._data_shape[0], self._mask]
        # Change the data type into uint16 if necessary
        if cropped.dtype == np.int16:
            # add 32768 to go from -32768-32767 to 0-65535
            cropped = cropped.view('uint16') + 0x8000
        elif cropped.dtype != np.uint16:
            raise ValueError('Only int16 and uint16 data types are supported at the moment.')

        if self._bidirectional:  # note: requires the mask to be symmetrical
            cropped[1::2, :] = cropped[1::2, ::-1]

        return cropped

    def _fetch(self) -> np.ndarray:  # noqa
        """Reads the acquired data from the input task."""
        if self._simulation is None:
            raw = self._read_task.in_stream.read()
            self._read_task.stop()
            self._write_task.stop()
        elif self._simulation == 'horizontal':
            raw = (self._x_axis._to_pos(self._scan_pattern[1, :] * u.V) * 10000).round().astype('int16')  # in mV
        elif self._simulation == 'vertical':
            raw = (self._y_axis._to_pos(self._scan_pattern[0, :] * u.V) * 10000).round().astype('int16')
        else:
            raise ValueError(
                f"Invalid simulation option {self._simulation}. Should be 'horizontal', 'vertical', or 'None'")

        # Preprocess raw data if a preprocess function is set
        if self._preprocess is None:
            preprocessed_raw = raw
        elif callable(self._preprocess):
            preprocessed_raw = self._preprocess(data=raw, sample_rate=self._sample_rate)
        else:
            raise TypeError(f"Invalid type for {self._preprocess}. Should be callable or None.")
        return self._raw_to_cropped(preprocessed_raw)

    def close(self):
        """Close connection to the NI-DAQ."""
        self._read_task.close()
        self._write_task.close()

    @property
    def preprocess(self):
        """The function to preprocess raw data before cropping."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: Optional[callable]):
        if not callable(value) and value is not None:
            raise TypeError(f"Invalid type for {self._preprocess}. Should be callable or None.")
        self._preprocess = value

    @property
    def pixel_size(self) -> Quantity:
        """The size of a pixel in the object plane."""
        extent_y = (self._y_axis.v_max - self._y_axis.v_min) * self._scale[0]
        extent_x = (self._x_axis.v_max - self._x_axis.v_min) * self._scale[1]
        return (Quantity(extent_y, extent_x) / self._base_resolution / (self._reference_zoom * self._zoom)).to(u.um)

    @property
    def duration(self) -> Quantity[u.ms]:
        """Total duration of scanning for one frame."""
        self._ensure_valid()
        return (self._scan_pattern.shape[1] / self._sample_rate).to(u.ms)

    @property
    def left(self) -> int:
        """The leftmost pixel of the Region of Interest (ROI) in the scan range."""
        return self._roi_left

    @left.setter
    def left(self, value: int):
        self._roi_left = int(value)
        self._valid = False

    @property
    def top(self) -> int:
        """The topmost pixel of the ROI in the scan range."""
        return self._roi_top

    @top.setter
    def top(self, value: int):
        self._roi_top = int(value)
        self._valid = False

    @property
    def height(self) -> int:
        """The number of pixels in the vertical dimension of the ROI."""
        return self._data_shape[0]

    @height.setter
    def height(self, value):
        self._data_shape = (int(value), int(self.data_shape[1]))
        self._valid = False

    @property
    def width(self) -> int:
        """The number of pixels in the horizontal dimension of the ROI."""
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._data_shape = (self.data_shape[0], int(value))
        self._valid = False

    @property
    def dwell_time(self) -> Quantity[u.us]:
        """The time spent on each pixel during scanning."""
        return (self._binning / self._sample_rate).to(u.us)

    @dwell_time.setter
    def dwell_time(self, value: Quantity[u.us]):
        self._binning = int(np.ceil(value * self._sample_rate))

    @property
    def delay(self) -> Quantity[u.us]:
        """Delay between the control signal to the mirrors and the start of data acquisition."""
        return self._delay  # add unit

    @delay.setter
    def delay(self, value: Quantity[u.us]):
        self._delay = value
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
        TODO: When the zoom factor is changed,
        the center of the region of interest and the number of pixels in the data remain constant.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        # roi_center = 0.5 * self._data_shape[1] + self._roi_start
        # roi_width_old = self._roi_end - self._roi_start
        # roi_width_new = roi_width_old * self._zoom / value
        # self._roi_start = roi_center - 0.5 * roi_width_new
        # self._roi_end = roi_center + 0.5 * roi_width_new
        self._zoom = float(value)
        self._valid = False

    @property
    def binning(self) -> int:
        """Undersampling factor.

        Increasing the binning reduces the number of pixels in the image while keeping dwell time the same.
        As a result, the total duration of a scan decreases.

        Note:
            This behavior is different from that of a real camera.
            No actual binning is performed, the scanner just takes fewer steps in x and y

        Note: the ROI is kept the same as much as possible.
            However, due to rounding, it may vary slightly.

        """
        return self._binning

    @binning.setter
    def binning(self, value: int):
        self._binning = int(value)
        self._valid = False

    @staticmethod
    def list_devices():
        return [d.name for d in nidaqmx.system.System().devices]
