from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ..core import Detector


class WFSResult:
    """
    Data structure for holding wavefront shaping results and statistics.

    Attributes:
        t (ndarray): Measured transmission matrix. If multiple targets were used, the first dimension(s) of `t`
            denote the columns of the transmission matrix (`a` indices), and the last dimensions(s) denote the targets,
            i.e., the rows of the transmission matrix (`b` indices).
        axis (int): Number of dimensions used for denoting a single column of the transmission matrix
            (e.g., 2 dimensions representing the x and y coordinates of the SLM pixels).
        noise_factor (ndarray): The estimated loss in fidelity caused by the limited SNR (for each target).
        amplitude_factor (ndarray): Estimated reduction of the fidelity due to phase-only modulation (for each target)
            (≈ π/4 for fully developed speckle).
        non_linearity (ndarray): Estimated deviation from a sinusoid response (TODO: experimental, untested).
        n (int): Total number of segments used in the optimization. When missing, this value is set to the number of
            elements in the first `axis` dimensions of `t`.
        estimated_optimized_intensity (ndarray): When missing, estimated intensity in the target(s) after displaying the
            wavefront correction on a perfect phase-only SLM.
        intensity_offset (Optional[ndarray]): Offset in the signal strength, as a scalar, or as one value per target.
            This is the offset that is caused by a bias in the detector signal, stray light, etc. Default value: 0.0.
    """

    def __init__(self,
                 t: np.ndarray,
                 axis: int,
                 noise_factor: ArrayLike,
                 amplitude_factor: ArrayLike,
                 non_linearity: ArrayLike,
                 n: Optional[int] = None,
                 intensity_offset: Optional[ArrayLike] = 0.0,
                 t_f: np.ndarray = None):
        """
        Args:
            t(ndarray): measured transmission matrix.
            axis(int):
                number of dimensions used for denoting a single columns of the transmission matrix
                (e.g. 2 dimensions representing the x and y coordinates of the SLM pixels)
            noise_factor(ArrayLike):
                the estimated loss in fidelity caused by the the limited snr (for each target).
            amplitude_factor(ArrayLike):
                estimated reduction of the fidelity due to phase-only modulation (for each target)
                (≈ π/4 for fully developed speckle)
            non_linearity(ArrayLike):
                estimated deviation from a sinusoid responds (TODO: experimental, untested)
            n(Optional[int]): total number of segments used in the optimization.
                when missing, this value is set to the number of elements in the first `axis` dimensions of `t`.
            intensity_offset(Optional[ArrayLike]):
                offset in the signal strength, as a scalar, or as one value per target
                this is the offset that is caused by a bias in the detector signal, stray light, etc.
                default value: 0.0

        """
        self.t = t
        self.t_f = t_f
        self.axis = axis
        self.noise_factor = np.atleast_1d(noise_factor)
        self.n = np.prod(t.shape[0:axis]) if n is None else n
        self.amplitude_factor = np.atleast_1d(amplitude_factor)
        self.estimated_enhancement = np.atleast_1d(1.0 + (self.n - 1) * self.amplitude_factor * self.noise_factor)
        self.non_linearity = np.atleast_1d(non_linearity)
        self.intensity_offset = intensity_offset * np.ones(self.non_linearity.shape) if np.isscalar(intensity_offset) \
            else intensity_offset
        after = np.sum(np.abs(t), tuple(range(self.axis))) ** 2 * self.noise_factor + intensity_offset
        self.estimated_optimized_intensity = np.atleast_1d(after)

    def __str__(self) -> str:
        noise_warning = "OK" if self.noise_factor > 0.5 else "WARNING low signal quality."
        amplitude_warning = "OK" if self.amplitude_factor > 0.5 else "WARNING uneven contribution of optical modes."
        non_linearity_warning = "OK" if self.non_linearity < 0.3 else ("WARNING non-linear phase response, check "
                                                                       "lookup table.")
        return f"""
        Wavefront shaping results:
            noise_factor: {self.noise_factor} {noise_warning}
            amplitude_factor: {self.amplitude_factor} {amplitude_warning}
            non_linearity: {self.non_linearity} {non_linearity_warning}
            estimated_enhancement: {self.estimated_enhancement}
            estimated_optimized_intensity: {self.estimated_optimized_intensity}
            """

    def select_target(self, b) -> 'WFSResult':
        """
        Returns the wavefront shaping results for a single target

        Args:
            b(int): target to select, as integer index.
                If the target array is multi-dimensional, it is flattened before selecting the `b`-th component.

        Returns: WFSResults data for the specified target
        """
        return WFSResult(t=self.t.reshape((*self.t.shape[0:2], -1))[:, :, b],
                         t_f=self.t_f.reshape((*self.t.shape[0:2], -1))[:, :, b],
                         axis=self.axis,
                         intensity_offset=self.intensity_offset[:][b],
                         noise_factor=self.noise_factor[:][b],
                         amplitude_factor=self.amplitude_factor[:][b],
                         non_linearity=self.non_linearity[:][b],
                         n=self.n,
                         )


def analyze_phase_stepping(measurements: np.ndarray, axis: int, A: Optional[float] = None):
    """Analyzes the result of phase stepping measurements, returning matrix `t` and noise statitics

    This function assumes that all measurements were made using the same reference field `A`
    and that the phase of the modulated segment/mode is phase-stepped in equally spaced steps
    between 0 and 2π, whereas the phase of the reference field is kept constant.

    Args:
        measurements(ndarray): array of phase stepping measurements.
            The array holds measured intensities
            with the first one or more dimensions corresponding to the segments(pixels) of the SLM,
            one dimension corresponding to the phase steps,
            and the last zero or more dimensions corresponding to the individual targets
            where the feedback was measured.
        axis(int): indicates which axis holds the phase steps.
        A(Optional[float]): magnitude of the reference field.
            This value is used to correctly normalize the returned transmission matrix.
            When missing, the value of `A` is estimated from the measurements.

    With `phase_steps` phase steps, the measurements are given by

    .. math::

        I_p = \\lvert A + B \\exp(i 2\\pi p / phase_{steps})\\rvert^2,

    This function computes the Fourier transform.

    .. math::

        \\frac{1}{phase_{steps}} \\sum I_p  \\exp(-i 2\\pi p / phase_{steps}) = A^* B

    The value of A^* B for each set of measurements is stored in the `field` attribute of the return
    value.
    Other attributes hold an estimate of the signal-to-noise ratio,
    and an estimate of the maximum enhancement that can be expected
    if these measurements are used for wavefront shaping.
    """
    phase_steps = measurements.shape[axis]
    n = int(np.prod(measurements.shape[:axis]))
    # M = np.prod(measurements.shape[axis + 1:])
    segments = tuple(range(axis))

    # Fourier transform the phase stepping measurements
    t_f = np.fft.fft(measurements, axis=axis) / phase_steps

    # the offset is the 0-th Fourier component
    # the signal is the first Fourier component (lowest non-zero frequency)
    # the -1 Fourier component is just the conjugate of the first Fourier component since the signal is real.
    # all other components should be 0. If they are not, they are considered to be noise

    I_f = np.sum(np.abs(t_f) ** 2, segments)  # total intensity per Fourier component per target
    signal_energy = I_f[1, ...]
    offset_energy = I_f[0, ...]
    total_energy = np.sum(I_f, axis=0)
    if phase_steps > 3:
        noise_energy = (total_energy - 2.0 * signal_energy - offset_energy) / (phase_steps - 3)
        noise_factor = np.maximum(signal_energy - noise_energy, 0.0) / signal_energy
    else:
        noise_factor = 1.0  # cannot estimate reliably

    if phase_steps > 6 and (I_f[2, ...] + I_f[3, ...]) != 0.0:
        non_linearity = (I_f[2, ...] - I_f[3, ...]) / (I_f[2, ...] + I_f[3, ...])
    else:
        non_linearity = 0.0  # cannot estimate reliably, or I_f[2] + I_f[3] == 0 (happens in simulation)

    if A is None:  # reference field strength not known: estimate from data
        t_abs = np.abs(np.take(t_f, 1, axis=axis))
        offset = np.take(t_f, 0, axis=axis)
        a_plus_b = np.sqrt(offset + 2.0 * t_abs)
        a_minus_b = np.sqrt(offset - 2.0 * t_abs)
        A = 0.5 * np.mean(a_plus_b + a_minus_b)

    t = np.take(t_f, 1, axis=axis) / A

    # compute the effect of amplitude variations.
    # for perfectly developed speckle, and homogeneous illumination, this factor will be pi/4
    amplitude_factor = np.mean(np.abs(t), segments) ** 2 / np.mean(np.abs(t) ** 2, segments)

    return WFSResult(t, t_f=t_f, axis=axis, amplitude_factor=amplitude_factor, noise_factor=noise_factor,
                     non_linearity=non_linearity, n=n)


class WFSController:
    """
    Controller for Wavefront Shaping (WFS) operations using a specified algorithm in the MicroManager environment.
    Manages the state of the wavefront and executes the algorithm to optimize and apply wavefront corrections, while
    exposing all these parameters to MicroManager.
    """

    class State(Enum):
        FLAT_WAVEFRONT = 0
        SHAPED_WAVEFRONT = 1

    def __init__(self, algorithm):
        """
        Args:
            algorithm: An instance of a wavefront shaping algorithm.
            source (Detector): An image source, e.g. a camera or scanner.
        """
        self.algorithm = algorithm
        self.source = source
        self._wavefront = WFSController.State.FLAT_WAVEFRONT
        self._result = None
        self._noise_factor = None
        self._amplitude_factor = None
        self._estimated_enhancement = None
        self._non_linearity = None
        self._estimated_optimized_intensity = None
        self._snr = None  # Average SNR. Computed when wavefront is computed.
        self._optimized_wavefront = None
        self._recompute_wavefront = False
        self._feedback_enhancement = None
        self._test_wavefront = False  # Trigger to test the optimized wavefront
        self._run_troubleshooter = False  # Trigger troubleshooter
        self.dark_frame = None
        self.before_frame = None

    @property
    def wavefront(self) -> State:
        """
        Gets the current wavefront state.

        Returns:
            State: The current state of the wavefront, either FLAT_WAVEFRONT or SHAPED_WAVEFRONT.
        """
        return self._wavefront

    @wavefront.setter
    def wavefront(self, value):
        """
        Sets the wavefront state and applies the corresponding phases to the SLM.

        Args:
            value (State): The desired state of the wavefront to set.
        """
        self._wavefront = value
        if value == WFSController.State.FLAT_WAVEFRONT:
            self.algorithm.slm.set_phases(0.0)
        else:
            if self._recompute_wavefront or self._optimized_wavefront is None:
                # select only the wavefront and statistics for the first target
                result = self.algorithm.execute().select_target(0)
                self._optimized_wavefront = -np.angle(result.t)
                self._noise_factor = result.noise_factor
                self._amplitude_factor = result.amplitude_factor
                self._estimated_enhancement = result.estimated_enhancement
                self._non_linearity = result.non_linearity
                self._estimated_optimized_intensity = result.estimated_optimized_intensity
                self._snr = 1.0 / (1.0 / result.noise_factor - 1.0)
                self._result = result
            self.algorithm.slm.set_phases(self._optimized_wavefront)

    @property
    def noise_factor(self) -> float:
        """
        Returns:
            float: noise factor: the estimated loss in fidelity caused by the the limited snr.
        """
        return self._noise_factor

    @property
    def amplitude_factor(self) -> float:
        """
        Returns:
            float: amplitude factor: estimated reduction of the fidelity due to phase-only
            modulation (≈ π/4 for fully developed speckle)
        """
        return self._amplitude_factor

    @property
    def estimated_enhancement(self) -> float:
        """
        Returns:
            float: estimated enhancement: estimated ratio <after>/<before>  (with <> denoting
            ensemble average)
        """
        return self._estimated_enhancement

    @property
    def non_linearity(self) -> float:
        """
        Returns:
            float: non-linearity.
        """
        return self._non_linearity

    @property
    def estimated_optimized_intensity(self) -> float:
        """
        Returns:
            float: estimated optimized intensity.
        """
        return self._estimated_optimized_intensity

    @property
    def snr(self) -> float:
        """
        Gets the signal-to-noise ratio (SNR) of the optimized wavefront.

        Returns:
            float: The average SNR computed during wavefront optimization.
        """
        return self._snr

    @property
    def recompute_wavefront(self) -> bool:
        """Returns: bool that indicates whether the wavefront needs to be recomputed. """
        return self._recompute_wavefront

    @recompute_wavefront.setter
    def recompute_wavefront(self, value):
        """Sets the bool that indicates whether the wavefront needs to be recomputed. """
        self._recompute_wavefront = value

    @property
    def feedback_enhancement(self) -> float:
        """Returns: the average enhancement of the feedback, returns none if no such enhancement was measured."""
        return self._feedback_enhancement

    @property
    def test_wavefront(self) -> bool:
        """Returns: bool that indicates whether test_wavefront will be performed if set."""
        return self._test_wavefront

    @test_wavefront.setter
    def test_wavefront(self, value):
        """
        Calculates the feedback enhancement between the flat and shaped wavefronts by measuring feedback for both
        cases.

        Args:
            value (bool): True to enable test mode, False to disable.
        """
        if value:
            self.wavefront = WFSController.State.FLAT_WAVEFRONT
            feedback_flat = self.algorithm.feedback.read().copy()
            self.wavefront = WFSController.State.SHAPED_WAVEFRONT
            feedback_shaped = self.algorithm.feedback.read().copy()
            self._feedback_enhancement = float(feedback_shaped.sum() / feedback_flat.sum())

        self._test_wavefront = value
