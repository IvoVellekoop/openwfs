from dataclasses import dataclass
from enum import Enum
from typing import Optional, Annotated, Union

import astropy.units as u
import numpy as np
from annotated_types import Ge, Le
from astropy.units import Quantity

from . import safe_import

ni = safe_import("nidaqmx", "nidaq")
if ni is not None:
    import nidaqmx.system
    from nidaqmx.constants import TerminalConfiguration, DigitalWidthUnits
    from nidaqmx.stream_writers import AnalogMultiChannelWriter

from ..core import Detector
from ..utilities import unitless


@dataclass
class InputChannel:
    """Specification of a NIDAQ input channel for the scanning microscope.

    Attributes:
        channel: The name of the channel, e.g. 'Dev4/ai0'
        v_min: The minimum voltage that can be measured by the channel
        v_max: The maximum voltage that can be measured by the channel
        terminal_configuration: The terminal configuration of the channel,
            defaults to `TerminalConfiguration.DEFAULT`
    """

    channel: str
    v_min: Quantity[u.V]
    v_max: Quantity[u.V]
    terminal_configuration: TerminalConfiguration = TerminalConfiguration.DEFAULT


@dataclass
class Axis:
    """Specification of a NIDAQ output channel for controlling a scan axis.

    Output voltages are clipped to the safety limit indicated by `[v_min, v_max]`
    Note that the actual field of view only covers part of this range (see
    `reference_zoom` in `ScanningMicroscope.`).

    Attributes:
        channel: The name of the channel, e.g. 'Dev4/ao0'
        v_min: The minimum voltage that can safely be sent to the output.
        v_max: The maximum voltage that can safely be sent to the output.
        scale: Conversion factor between voltage at the NI-DAQ card and displacement of the focus in the object plane.
                This may be different for different axes.

        maximum_acceleration: The maximum acceleration of this axis in V per s².
            The output signal will be constructed to ensure that the mirror does not exceed this acceleration,
            except in special cases such as when turning on the system. This acceleration is also used to compute
            how far to overshoot the scan mirror to ensure that it reaches a linear speed over the scan range.
            See `scan` for details.
        terminal_configuration: The terminal configuration of the channel,
            defaults to `TerminalConfiguration.DEFAULT`
    """

    channel: str
    v_min: Quantity[u.V]
    v_max: Quantity[u.V]
    scale: Quantity[u.um / u.V]
    maximum_acceleration: Quantity[u.V / u.s**2]
    terminal_configuration: TerminalConfiguration = TerminalConfiguration.DEFAULT

    def to_volt(self, pos: Union[np.ndarray, float]) -> Quantity[u.V]:
        """Converts relative position [0.0 ... 1.0] to voltage [V_min ... V_max]

        Currently, this is just a linear conversion, but a lookup table may be used in the future.
        """
        return self.v_min + np.clip(pos, 0.0, 1.0) * (self.v_max - self.v_min)

    def to_pos(self, volt: Quantity[u.V]) -> np.ndarray:
        """Converts voltage [V_min .. V_max] to relative position [0.0 .. 1.0]"""
        return unitless((volt - self.v_min) / (self.v_max - self.v_min))

    def maximum_scan_speed(self, linear_range: float):
        """Computes the maximum scan speed in V per sample

        It is assumed that the mirror accelerates and decelerates at the maximum
        acceleration, and scans with a constant velocity over the linear range.
        There are two limits to the scan speed:

        - A practical limit: if it takes longer to perform the acceleration + deceleration than
          it does to traverse the linear range, it does not make sense to set the scan speed so high.
          The speed at which acceleration + deceleration takes as long as the linear range is the maximum speed.
        - A hardware limit: when accelerating with the maximum acceleration over a distance
          0.5 · (V_max-V_min) · (1-linear_range),
          the mirror will reach the maximum possible speed.

        Args:
            linear_range (float): fraction of the full range that is used for the linear part of the scan

        Returns:
            Quantity[u.V / u.s]: maximum scan speed

        """
        # x = 0.5 · a · t² = 0.5 (v_max - v_min) · (1 - linear_range)
        t_accel = np.sqrt((self.v_max - self.v_min) * (1 - linear_range) / self.maximum_acceleration)
        hardware_limit = t_accel * self.maximum_acceleration

        # t_linear = linear_range · (v_max - v_min) / maximum_speed
        # t_accel = maximum_speed / maximum_acceleration
        # 0.5·t_linear == t_accel => 0.5·linear_range · (v_max-v_min) · maximum_acceleration = maximum_speed²
        practical_limit = np.sqrt(0.5 * linear_range * (self.v_max - self.v_min) * self.maximum_acceleration)
        return np.minimum(hardware_limit, practical_limit)

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
        if start == stop:
            return np.zeros((0,)) * u.V  # return empty array
        v_end = self.to_volt(stop)

        # `t` is measured in samples
        # `a` is measured in volt/sample²
        a = self.maximum_acceleration / sample_rate**2 * np.sign(v_end - v_start)
        t_total = unitless(2.0 * np.sqrt((v_end - v_start) / a))
        t = np.arange(np.ceil(t_total + 1e-6))  # add a small number to deal with case t=0 (start=end)
        v_accel = v_start + 0.5 * a * t[: len(t) // 2] ** 2  # acceleration part
        v_decel = v_end - 0.5 * a * (t_total - t[len(t) // 2 :]) ** 2  # deceleration part
        v_decel[-1] = v_end  # fix last point because t may be > t_total due to rounding
        return np.clip(np.concatenate((v_accel, v_decel)), self.v_min, self.v_max)  # noqa ignore incorrect type warning

    def scan(self, start: float, stop: float, sample_count: int, sample_rate: Quantity[u.Hz]):
        """
        Generate a voltage sequence to scan with a constant velocity from start to stop,
        including acceleration and deceleration.

        Before starting this sequence, the mirror is assumed to be standing still at the launch point,
        which is some distance _before_ start.
        After the scan sequence, the mirror is stopped at the landing point,
        which is some distance _after_ stop.
        The launch point and landing point are returned along with the scan sequence.

        This function also returns a slice object, which represents the part of the sequence
        that corresponds to a linear movement from start to stop. ``slice.stop - slice.start = sample_count``.

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
        if start == stop:  # todo: tolerance?
            return (
                np.ones((sample_count,)) * v_start,
                start,
                start,
                slice(0, sample_count),
            )

        v_end = self.to_volt(stop)
        scan_speed = (v_end - v_start) / sample_count  # V per sample

        # construct a sequence to accelerate from speed 0 to the scan speed
        # we start by constructing a sequence with a maximum acceleration.
        # This sequence may be up to 1  sample longer than needed to reach the scan speed.
        # This last sample is replaced by movement at a linear scan speed
        a = self.maximum_acceleration / sample_rate**2 * np.sign(scan_speed)  # V per sample²
        t_launch = np.arange(np.ceil(unitless(scan_speed / a)))  # in samples
        v_accel = 0.5 * a * t_launch**2  # last sample may have faster scan speed than needed
        if len(v_accel) > 1 and np.abs(v_accel[-1] - v_accel[-2]) > np.abs(scan_speed):
            v_accel[-1] = v_accel[-2] + scan_speed
        v_launch = v_start - v_accel[-1] - 0.5 * scan_speed  # launch point
        v_land = v_end + v_accel[-1] + 0.5 * scan_speed  # landing point

        # linear part of the scan
        v_linear = v_start + scan_speed * (np.arange(sample_count) + 0.5)

        # combine the parts
        v = np.concatenate((v_launch + v_accel, v_linear, v_land - v_accel[::-1]))
        v = np.clip(v, self.v_min, self.v_max)
        launch = self.to_pos(v_launch)
        land = self.to_pos(v_land)
        return v, launch, land, slice(len(v_accel), len(v_accel) + sample_count)

    @staticmethod
    def compute_scale(
        *,
        optical_deflection: Quantity[u.deg / u.V],
        galvo_to_pupil_magnification: float,
        objective_magnification: float,
        reference_tube_lens: Quantity[u.mm],
    ) -> Quantity[u.um / u.V]:
        """Computes the conversion factor between voltage and displacement in the object plane.

        Args:
            optical_deflection (Quantity[u.deg/u.V]):
                The optical deflection (i.e. twice the mechanical angle) of the mirror
                 as a function of applied voltage.
            galvo_to_pupil_magnification (float):
                The magnification of the relay system between the galvo mirrors and the pupil.
            objective_magnification (Quantity[u.mm]):
                The magnification of the microscope objective.
            reference_tube_lens (Quantity[u.mm]):
                The tube lens focal length on which the objective magnification is based.
                This value is manufacturer-specific. Typical values are:
                - 200 mm for Thorlabs, Nikon, Leica, and Mitutoyo
                - 180 mm for Olympus/Evident
                - 165 mm for Zeiss

        Returns:
            Quantity[u.um/u.V]: The conversion factor between voltage and displacement in the object plane.
        """
        f_objective = reference_tube_lens / objective_magnification
        angle_to_displacement = f_objective / u.rad
        return ((optical_deflection / galvo_to_pupil_magnification) * angle_to_displacement).to(u.um / u.V)

    @staticmethod
    def compute_acceleration(
        *,
        optical_deflection: Quantity[u.deg / u.V],
        torque_constant: Quantity[u.N * u.m / u.A],
        rotor_inertia: Quantity[u.kg * u.m**2],
        maximum_current: Quantity[u.A],
    ) -> Quantity[u.V / u.s**2]:
        """Computes the angular acceleration of the focus of the galvo mirror.

         The result is returned in the unit V / second²,
         where the voltage can be converted to displacement using the scale factor.

        Args:
            optical_deflection (Quantity[u.deg/u.V]):
                The optical deflection (i.e. twice the mechanical angle) of the mirror
                 as a function of applied voltage.
            torque_constant (Quantity[u.N*u.m/u.A]):
                The torque constant of the galvo mirror driving coil.
                May also be given in the equivalent unit of dyne·cm/A.
            rotor_inertia (Quantity[u.kg*u.m**2]):
                The moment of inertia of the rotor. May also be given in the equivalent unit of g·cm².
            maximum_current (Quantity[u.A]):
                The maximum current that can be applied to the galvo mirror.
        """
        angular_acceleration = (torque_constant * maximum_current / rotor_inertia).to(u.s**-2) * u.rad
        return (angular_acceleration / optical_deflection).to(u.V / u.s**2)


class TestPatternType(Enum):
    """Type of test pattern to use for simulation."""

    NONE = "none"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    IMAGE = "image"


class ScanningMicroscope(Detector):
    """Laser scanning microscope with galvo mirrors controlled by a National Instruments data acquisition card (nidaq).

    Effectively, a `ScanningMicroscope` works like a camera, which can be triggered and returns 2-D images.
    These images are obtained by raster-scanning a focus using two galvo mirrors, controlled by a nidaq card,
    and recording a detector signal (typically from a photon multiplier tube (PMT)) with the same card.

    Region of Interest (ROI):
        Upon construction, the maximum voltage range is set for both axes.
        To avoid possible damage to the hardware, this range cannot be exceeded during scanning,
        and the value cannot be changed after construction of the ScanningMicroscope.
        It is recommended to set this voltage range slightly larger than the maximum field of view
        of the microscope (but still within the limits that may cause damage to the hardware),
        so that the mirror has some room to accelerate and decelerate at the edges of the scan range.

        The scan region is defined by the following equation:

        V_start = V_min + (center - 0.5 / (reference_zoom * zoom)) * (V_max - V_min)
        V_stop = V_min + (center + 0.5 / (reference_zoom * zoom)) * (V_max - V_min)

        or, in relative coordinates:

        start_full = center - 0.5 / (reference_zoom * zoom)
        stop_full = center + 0.5 / (reference_zoom * zoom)

        with
        * V_min, V_max: The voltage ranges set for both axes:
        * zoom, reference_zoom: The zoom factors
        * center: The center of the image relative to the full voltage range, default value is 0.5

        The scan region is divided into `resolution × resolution` pixels.
        Within this scan region, a smaller region of interest (ROI) can be defined by setting
        `top`, `left`, `height`, and `width` properties. The ROI is defined in pixels,
        with `(0,0,resolution, resolution)` corresponding to the full field of view.

        start = center + (left / resolution - 0.5) / (reference_zoom * zoom)
        stop = center + ((left + width) / resolution - 0.5) / (reference_zoom * zoom)

    Scan pattern:
        The scanner performs a raster scan, with `y` being the slow axis and `x` the fast axis.
        The pattern is computed such that the mirrors have a constant velocity during the scan.
        The start and end of each scan line, where the mirror accelerates or decelerates are discarded from the data.
        By default, the scanner uses bidirectional scanning along the fast axis,
        which reduces the time needed for a full scan.
        Especially for bidirectional scanning, the synchronization between output and input is crucial, otherwise the
        image will appear teared (even and odd scan lines not aligning). To fine-tune this synchronization, the `delay`
        parameter can be used.
    """

    def __init__(
        self,
        input: InputChannel,
        y_axis: Axis,
        x_axis: Axis,
        sample_rate: Quantity[u.MHz],
        resolution: int,
        reference_zoom: float,
        *,
        delay: Quantity[u.us] = 0.0 * u.us,
        bidirectional: bool = True,
        multi_threaded: bool = True,
        preprocessor: Optional[callable] = None,
        test_pattern: Union[TestPatternType, str] = TestPatternType.NONE,
        test_image=None,
    ):
        """
        Args:
            resolution: number of pixels (height and width) in the full field of view.
                A coarser sampling can be achieved by setting the binning
                Note that the ROI can also be reduced by setting width, height, top and left.
            input: The NI-DAQ channel to use for the input.
            y_axis: The scan axis object for controlling the slow axis.
            x_axis: The scan axis object for controlling the fast axis.
            sample_rate:
                Sample rate of the NI-DAQ input channel.
            delay: Delay between mirror control and data acquisition, measured in microseconds
            reference_zoom: Zoom factor that corresponds to fitting the full field of view exactly.
                The zoom factor in the `zoom` property is multiplied by the `reference_zoom` to compute the scan range.
            bidirectional: If true, enables bidirectional scanning along the fast axis.
            preprocessor: Process the raw data with this function before cropping. When None, the
                preprocessing will be skipped. The function must take input arguments data and sample_rate, and must
                return the preprocessed data.
            test_pattern: Type of test pattern to use for simulation. When set to a value other than 'none', the nidaq hardware is bypassed
                completely, and a test pattern displayed, depending on the value of this parameter:
                - 'horizontal': The voltage that would be sent to the fast axis output channel is used as input value.
                - 'vertical': The voltage that would be sent to the slow axis output channel is used as input value.
                - 'image': The voltages that would be sent are converted to coordinates in an image, resulting in the test image to be returned.
            test_image: The test image to use when `test_pattern` is set to 'image'. This image is expected to be a 2D numpy array
        """
        self._y_axis = y_axis
        self._x_axis = x_axis
        self._input_channel = input
        self._sample_rate = sample_rate.to(u.MHz)
        self._binning = 1  # binning factor
        self._resolution = int(resolution)
        self._roi_top = 0  # in pixels
        self._roi_left = 0  # in pixels
        self._center_x = 0.5  # in relative coordinates (relative to the full field of view)
        self._center_y = 0.5  # in relative coordinates (relative to the full field of view)
        self._delay = delay.to(u.us)
        self._reference_zoom = float(reference_zoom)
        self._zoom = 1.0
        self._bidirectional = bool(bidirectional)
        self._oversampling = 1  # oversampling factor
        self._scan_speed_factor = 0.5  # scan speed relative to maximum

        self._test_pattern = TestPatternType(test_pattern)
        self._test_image = None
        if test_image is not None:
            self._test_image = np.array(test_image, dtype="uint16")
            while self._test_image.ndim > 2:
                self._test_image = np.mean(self._test_image, 2).astype("uint16")

        self._preprocessor = preprocessor

        self._write_task = None
        self._read_task = None

        self._valid = False  # indicates that `trigger()` should initialize the NI-DAQ tasks and scan pattern
        self._scan_pattern = None

        # the pixel size and duration are computed dynamically
        # data_shape just returns self._data shape, and latency = 0.0 ms
        super().__init__(
            data_shape=(resolution, resolution),
            pixel_size=None,
            duration=None,
            latency=0.0 * u.ms,
            multi_threaded=multi_threaded,
        )
        self._update()

    def _update(self):
        """Computes the scan pattern"""

        width = self._data_shape[1]
        height = self._data_shape[0]
        center = 0.5 * self._resolution

        # compute the size of a pixel relative to the maximum voltage range
        actual_zoom = self._reference_zoom * self._zoom
        roi_scale = 1.0 / actual_zoom / self._resolution

        roi_left = self._center_x + (self._roi_left - center) * roi_scale
        roi_right = self._center_x + (self._roi_left + width - center) * roi_scale
        roi_top = self._center_y + (self._roi_top - center) * roi_scale
        roi_bottom = self._center_y + (self._roi_top + height - center) * roi_scale

        # special case for roi width or height of 1.
        # Treat as roi size zero so that the beam does not scan across this single pixel.
        # note: this special handling is not needed for the height, because
        # in the vertical direction the beam does not scan over the pixel,
        # it just steps to the next line at the end of the line
        if width == 1:
            roi_right = 0.5 * (roi_left + roi_right)
            roi_left = roi_right

        # Compute the retrace pattern for the slow axis
        # The scan starts at half a pixel after roi_bottom and ends half a pixel before roi_top
        v_yr = self._y_axis.step(roi_bottom - 0.5 * roi_scale, roi_top + 0.5 * roi_scale, self._sample_rate)

        # Compute the scan pattern for the fast axis
        # The naive speed is the scan speed assuming one pixel per sample
        # The maximum speed is the maximum speed that the mirror can achieve over the scan range
        # (at least, without spending more time on accelerating and decelerating than the scan itself)
        # The user can set the scan speed relative to the maximum speed.
        # If this set speed is lower than naive scan speed, multiple samples are taken per pixel.
        naive_speed = (self._x_axis.v_max - self._x_axis.v_min) * roi_scale * self._sample_rate
        max_speed = self._x_axis.maximum_scan_speed(1.0 / actual_zoom) * self._scan_speed_factor
        if max_speed == 0.0:
            # this may happen if the ROI reaches to or beyond [0,1]. In this case, the mirror has no time to accelerate
            # TODO: implement an auto-adjust option instead of raising an error
            raise ValueError(
                "Maximum scan speed is zero. "
                "This may be because the region of interest exceeds the maximum voltage range"
            )

        self._oversampling = int(np.ceil(unitless(naive_speed / max_speed)))
        oversampled_width = width * self._oversampling
        v_x_even, x_launch, x_land, self._mask = self._x_axis.scan(
            roi_left, roi_right, oversampled_width, self._sample_rate
        )
        if self._bidirectional:
            v_x_odd, _, _, _ = self._x_axis.scan(roi_right, roi_left, oversampled_width, self._sample_rate)
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
        n_rows = self._data_shape[0] + np.ceil(len(v_yr) / len(v_x_odd)).astype("int32")
        self._n_cols = len(v_x_odd)
        if self._bidirectional and n_rows % 2 == 1:
            n_rows += 1

        scan_pattern = np.zeros((2, n_rows, self._n_cols))
        scan_pattern[1, 0::2, :] = v_x_even  # .reshape((1, -1))
        scan_pattern[1, 1::2, :] = v_x_odd

        y_coord = (np.arange(height) + 0.5) * roi_scale + roi_top
        scan_pattern[0, :height, :] = self._y_axis.to_volt(y_coord).reshape(-1, 1)

        # The last row(s) are used for the vertical retrace
        # We park the mirror after the vertical retrace, to allow
        # the horizontal scan mirror to finish its scan.
        # Note: this may not always be needed, but it guarantees
        # that the horizontal scan mirror is always scanning at the same frequency
        # which is essential for resonant scanning.
        if len(v_yr) > 0:
            retrace = scan_pattern[0, height:, :].reshape(-1)
            retrace[0 : len(v_yr)] = v_yr
            retrace[len(v_yr) :] = v_yr[-1]

        self._scan_pattern = scan_pattern.reshape(2, -1)
        self._valid = True  # indicate that scan patterns have been computed
        if self._test_pattern != TestPatternType.NONE:
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
        # set the timeout for the nidaq task. Note that this line calls .duration, which
        # in turn calls ._ensure_valid. Therefore, it is important that we gave set self._valid = True
        # above, to avoid an infinite loop.
        self._read_task.in_stream.timeout = self.timeout.to_value(u.s)

        # Configure the sample clock task
        sample_rate = self._sample_rate.to_value(u.Hz)
        sample_count = self._scan_pattern.shape[1]

        # Configure the analog output task (two channels)
        self._write_task.ao_channels.add_ao_voltage_chan(
            self._x_axis.channel,
            min_val=self._x_axis.v_min.to_value(u.V),
            max_val=self._x_axis.v_max.to_value(u.V),
        )
        self._write_task.ao_channels.add_ao_voltage_chan(
            self._y_axis.channel,
            min_val=self._y_axis.v_min.to_value(u.V),
            max_val=self._y_axis.v_max.to_value(u.V),
        )
        self._write_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)

        # Configure the analog input task (one channel)
        self._read_task.ai_channels.add_ai_voltage_chan(
            self._input_channel.channel,
            min_val=self._input_channel.v_min.to_value(u.V),
            max_val=self._input_channel.v_max.to_value(u.V),
            terminal_config=self._input_channel.terminal_configuration,
        )
        self._read_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=sample_count)
        self._read_task.triggers.start_trigger.cfg_dig_edge_start_trig(self._write_task.triggers.start_trigger.term)
        delay = self._delay.to_value(u.s)
        if delay > 0.0:
            self._read_task.triggers.start_trigger.delay = delay
            self._read_task.triggers.start_trigger.delay_units = DigitalWidthUnits.SECONDS

        self._writer = AnalogMultiChannelWriter(self._write_task.out_stream)

    def _ensure_valid(self):
        if not self._valid:
            self._update()

    def _do_trigger(self):
        """Makes sure scan patterns are up-to-date, and triggers the NI-DAQ tasks."""
        self._ensure_valid()

        if self._test_pattern != TestPatternType.NONE:
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
        cropped = raw.reshape(-1, self._n_cols)[: self._data_shape[0], self._mask]

        # down sample along fast axis if needed
        if self._oversampling > 1:
            # remove samples if not divisible by oversampling factor
            cropped = cropped[:, : (cropped.shape[1] // self._oversampling) * self._oversampling]
            cropped = cropped.reshape(cropped.shape[0], -1, self._oversampling)
            cropped = np.round(np.mean(cropped, 2)).astype(cropped.dtype)  # todo: faster alternative?

        # Change the data type into uint16 if necessary
        if cropped.dtype == np.int16:
            # add 32768 to go from -32768-32767 to 0-65535
            cropped = cropped.view("uint16") + 0x8000
        elif cropped.dtype != np.uint16:
            raise ValueError(f"Only int16 and uint16 data types are supported at the moment, got type {cropped.dtype}.")

        if self._bidirectional:  # note: requires the mask to be symmetrical
            cropped[1::2, :] = cropped[1::2, ::-1]

        return cropped

    def _fetch(self) -> np.ndarray:  # noqa
        """Reads the acquired data from the input task."""
        if self._test_pattern is TestPatternType.NONE:
            raw = self._read_task.in_stream.read()
            self._read_task.stop()
            self._write_task.stop()
        elif self._test_pattern == TestPatternType.HORIZONTAL:
            raw = np.round(self._x_axis.to_pos(self._scan_pattern[1, :] * u.V) * 10000).astype("int16")
        elif self._test_pattern == TestPatternType.VERTICAL:
            raw = np.round(self._y_axis.to_pos(self._scan_pattern[0, :] * u.V) * 10000).astype("int16")
        elif self._test_pattern == TestPatternType.IMAGE:
            if self._test_image is None:
                raise ValueError("No test image was provided for the image simulation.")
            # todo: cache the test image
            row = np.floor(
                self._y_axis.to_pos(self._scan_pattern[0, :] * u.V) * (self._test_image.shape[0] - 1)
            ).astype("int32")
            column = np.floor(
                self._x_axis.to_pos(self._scan_pattern[1, :] * u.V) * (self._test_image.shape[1] - 1)
            ).astype("int32")
            raw = self._test_image[row, column]
        else:
            raise ValueError(
                f"Invalid simulation option {self._test_pattern}. "
                "Should be 'horizontal', 'vertical', 'image', or 'None'"
            )

        # Preprocess raw data if a preprocess function is set
        if self._preprocessor is None:
            preprocessed_raw = raw
        elif callable(self._preprocessor):
            preprocessed_raw = self._preprocessor(data=raw, sample_rate=self._sample_rate)
        else:
            raise TypeError(f"Invalid type for {self._preprocessor}. Should be callable or None.")
        return self._raw_to_cropped(preprocessed_raw)

    def close(self):
        """Close connection to the NI-DAQ."""
        self._read_task.close()
        self._write_task.close()

    @property
    def test_pattern(self) -> TestPatternType:
        return self._test_pattern

    @test_pattern.setter
    def test_pattern(self, value: TestPatternType):
        self._test_pattern = TestPatternType(value)

    @property
    def preprocessor(self):
        """An optional function to preprocess raw data before cropping.

        The function takes a linear array of raw data as required arguments,
         and a list of keyword arguments. Currently, the following arguments are passed:
            - sample_rate (Quantity[u.MHz]): the sample rate of the NI-DAQ input channel
        """
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Optional[callable]):
        if not callable(value) and value is not None:
            raise TypeError(f"Invalid type for {self._preprocessor}. Should be callable or None.")
        self._preprocessor = value

    @property
    def pixel_size(self) -> Quantity:
        """The size of a pixel in the object plane."""
        # TODO: make extent a read-only attribute of Axis
        extent_y = (self._y_axis.v_max - self._y_axis.v_min) * self._y_axis.scale
        extent_x = (self._x_axis.v_max - self._x_axis.v_min) * self._x_axis.scale
        return (Quantity(extent_y, extent_x) / (self._reference_zoom * self._zoom * self._resolution)).to(u.um)

    @property
    def duration(self) -> Quantity[u.ms]:
        """Total duration of scanning for one frame."""
        self._ensure_valid()  # make sure _scan_pattern is up-to-date
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
        if value < 1 or value > self._resolution:
            raise ValueError(f"Height must be between 1 and {self._resolution}")
        self._data_shape = (int(value), int(self.data_shape[1]))
        self._valid = False

    @property
    def width(self) -> int:
        """The number of pixels in the horizontal dimension of the ROI.

        Depending on the scan speed and sample rate, the scanner may
        acquire multiple data points along a scan line, and return the
        averaged value.
        A value of 1 is treated as a special case,
        where the beam does not move horizontally.at all
        (i.e. it does not scan back and forth over the size of this single pixel).
        """
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        if value < 1 or value > self._resolution:
            raise ValueError(f"Width must be between 1 and {self._resolution}")
        self._data_shape = (self.data_shape[0], int(value))
        self._valid = False

    def reset_roi(self):
        """Reset the ROI to span the original left, top, width and height."""
        self.left = 0
        self.top = 0
        self.width = self._resolution
        self.height = self._resolution

    @property
    def dwell_time(self) -> Quantity[u.us]:
        """The time spent on each pixel during scanning."""
        return (self._oversampling / self._sample_rate).to(u.us)

    @property
    def delay(self) -> Quantity[u.us]:
        """Delay between the control signal to the mirrors and the start of data acquisition."""
        return self._delay  # add unit

    @delay.setter
    def delay(self, value: Quantity[u.us]):
        self._delay = value
        self._valid = False

    @property
    def exposure(self) -> Quantity[u.ms]:
        """The required time to scan a frame."""
        return self.duration

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
        When zooming in or out, the center of the region of interest is kept constant.
        Note that this may cause the field of view to get extended to outside the original FOV
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        # compute how far the roi center is away from the _center_x before the zoom
        roi_scale = 1.0 / (self._reference_zoom * self._zoom) / self._resolution
        center_y_before = (self._roi_top + 0.5 * self._data_shape[0]) * roi_scale
        center_x_before = (self._roi_left + 0.5 * self._data_shape[1]) * roi_scale

        # compute how far it will be from _center_x after adjusting the zoom
        center_y_after = center_y_before * self._zoom / value
        center_x_after = center_x_before * self._zoom / value

        # correct the center position such that the center of the roi does not move
        self._center_y += center_y_before - center_y_after
        self._center_x += center_x_before - center_x_after

        self._zoom = float(value)
        self._valid = False

    @property
    def offset_x(self) -> float:
        """The center of the full field of view in the horizontal direction.

        The offset is relative to the full voltage range specified in the Axis objects,
        with 0.0 corresponding to the center of the voltage range,
        and -0.5 and +0.5 to the edges of the voltage range.

        Note that changing the offset may cause the ROI to move outside the original field of view.
        Also, it may cause the scan speed to change, as the mirror has a shorter distance to accelerate or decelerate.
        """
        return self._center_x - 0.5

    @offset_x.setter
    def offset_x(self, value: float):
        self._center_x = float(value) + 0.5
        self._valid = False

    @property
    def offset_y(self) -> float:
        """The center of the full field of view in the vertical direction.

        The offset is relative to the full voltage range specified in the Axis objects,
        with 0.0 corresponding to the center of the voltage range,
        and -0.5 and +0.5 to the edges of the voltage range.

        Note that changing the offset may cause the ROI to move outside the original field of view.
        Also, it may cause the scan speed to change, as the mirror has a shorter distance to accelerate or decelerate.
        """
        return self._center_y - 0.5

    @offset_y.setter
    def offset_y(self, value: float):
        self._center_y = float(value) + 0.5
        self._valid = False

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, value: int):
        self._scale_roi(value / self._resolution)

    def _scale_roi(self, factor: float):
        """Adjusts resolution, top, left, width and height by the same factor."""

        def adjust(x):
            return int(np.round(x * factor))

        self._roi_left = adjust(self._roi_left)
        self._roi_top = adjust(self._roi_top)
        self._data_shape = (adjust(self._data_shape[0]), adjust(self._data_shape[1]))
        self._resolution = adjust(self._resolution)
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
        if value < 1:
            raise ValueError("Binning value should be a positive integer")
        self._scale_roi(self._binning / int(value))
        self._binning = int(value)

    @property
    def scan_speed(self) -> Annotated[float, Ge(0.05), Le(1.0)]:
        """The scan speed relative to the maximum scan speed."""
        return self._scan_speed_factor

    @scan_speed.setter
    def scan_speed(self, value):
        self._scan_speed_factor = np.clip(float(value), 0.05, 1.0)

    @staticmethod
    def list_devices():
        """Returns a list of all nidaq devices available on the system."""
        return [d.name for d in nidaqmx.system.System().devices]
