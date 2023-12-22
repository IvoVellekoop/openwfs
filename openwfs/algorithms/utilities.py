import numpy as np
from enum import Enum


class WFSResult:
    """
    Data structure for holding wavefront shaping results and statistics.

    Attributes:
        t: measured transmission matrix.
            if multiple targets were used, the first dimension(s) of t denote the columns of the transmission matrix
            (`a` indices) and the last dimensions(s) denote the targets, i.e., the rows of the transmission matrix
            (`b` indices).
        n: number of degrees of freedom (columns of the transmission matrix)
        snr: estimated signal-to-noise ratio for each of the targets.
        noise_factor: the estimated loss in fidelity caused by the the limited snr.
        amplitude_factor: estimated reduction of the fidelity due to phase-only modulation (≈ π/4 for fully developed
            speckle)
        estimated_improvement: estimated ratio after/before
        estimated_enhancement: estimated ratio <after>/<before>  (with <> denoting ensemble average)
    """

    def __init__(self, t, axis, noise_factor, amplitude_factor, non_linearity, n=None, I_offset=0.0):
        """
        Args:
            t: measured transmission matrix.
            snr: estimated signal-to-noise ratio for each of the targets.
                Used to compute the noise factor.
            amplitude_factor:
                estimated reduction of the fidelity due to phase-only modulation (≈ π/4 for fully developed speckle)
            estimated_improvement: estimated ratio before/after
        """
        self.t = t
        self.axis = axis
        self.noise_factor = np.atleast_1d(noise_factor)
        self.n = t.size / self.noise_factor.size if n is None else n
        self.amplitude_factor = np.atleast_1d(amplitude_factor)
        self.estimated_enhancement = np.atleast_1d(1.0 + (self.n - 1) * self.amplitude_factor * self.noise_factor)
        self.I_offset = np.atleast_1d(I_offset)
        self.non_linearity = np.atleast_1d(non_linearity)
        after = np.sum(np.abs(t), tuple(range(self.axis))) ** 2 * self.noise_factor + I_offset
        self.estimated_optimized_intensity = np.atleast_1d(after)

    def select_target(self, b) -> 'WFSResult':
        """
        Returns the wavefront shaping results for a single target

        Args:
            b(int): target to select, as integer index.
                If the target array is multi-dimensional, it is flattened before selecting the `b`-th component.

        Returns: WFSResults data for the specified target
        """
        return WFSResult(t=self.t.reshape((*self.t.shape[0:2], -1))[:, :, b],
                         axis=self.axis,
                         I_offset=self.I_offset[:][b],
                         noise_factor=self.noise_factor[:][b],
                         amplitude_factor=self.amplitude_factor[:][b],
                         non_linearity=self.non_linearity[:][b],
                         n=self.n
                         )


def analyze_phase_stepping(measurements: np.ndarray, axis: int, A=None):
    """Analyzes the result of phase stepping measurements, returning matrix `t` and noise statitics

    This function assumes that all measurements were made using the same reference field `A`
    and that the phase of the modulated segment/mode is phase-stepped in equally spaced steps
    between 0 and 2π.

    Args:
        measurements(ndarray): array of phase stepping measurements.
            The array holds measured intensities
            with the first one or more dimensions corresponding to the segments(pixels) of the SLM,
            one dimension corresponding to the phase steps,
            and the last zero or more dimensions corresponding to the individual targets
            where the feedback was measured.
        axis(int): indicates which axis holds the phase steps.

    With `phase_steps` phase steps, the measurements are given by

    .. math::

        I_p = \lvert A + B \\exp(i 2\pi p / phase_{steps})\\rvert^2,

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
    N = np.prod(measurements.shape[:axis])
    M = np.prod(measurements.shape[axis + 1:])
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

    # # determine scaling and offset of t
    # y = np.take(t_f, 0, axis=axis)  # this is |A|^2 + |B_i|^2 + C
    # # for SSA only:
    # y = y - 2 * np.real(np.take(t_f, 1, axis=axis))
    # x = np.abs(np.take(t_f, 1, axis=axis)) ** 2  # this is |A|^2|B_i|^2
    #
    # # linear fit
    # a = np.sum(x ** 2, segments)
    # b = np.sum(x, segments)
    # c = -np.sum(x * y, segments)
    # d = np.sum(y, segments)
    # alpha = -(b * d / N + c) / (a - b ** 2 / N)
    # I_A = 1.0 / alpha
    # I_A = np.mean(measurements, axis=segments)[0, ...]
    # I_offset = 0.0  # (d - alpha * b) / N - I_A
    # # print(f"residual {np.linalg.norm(x * alpha + I_offset + I_A - y)}")

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

    return WFSResult(t, axis=axis, amplitude_factor=amplitude_factor, noise_factor=noise_factor,
                     non_linearity=non_linearity)


class WFSController:
    """
    Controller for Wavefront Shaping (WFS) operations using a specified algorithm in the MicroManager environment.
    Manages the state of the wavefront and executes the algorithm to optimize and apply wavefront corrections, while
    exposing all these parameters to MicroManager.

    Attributes:
        Public:
            wavefront (State): Current state of the wavefront.
            recompute_wavefront (bool): Flag to indicate if the wavefront needs to be recomputed.
            feedback_enhancement (float): Measured feedback enhancement.
            test_wavefront (bool): Flag indicating if the wavefront is in test mode.
            snr (float): Average signal-to-noise ratio computed during wavefront optimization.
            algorithm: The wavefront shaping algorithm instance.

        Private:
            _optimized_wavefront (numpy.ndarray): Optimized wavefront data.
    """

    class State(Enum):
        FLAT_WAVEFRONT = 0
        SHAPED_WAVEFRONT = 1

    def __init__(self, algorithm):
        """
        Args:
            algorithm: An instance of a wavefront shaping algorithm.
        """
        self.algorithm = algorithm
        self._wavefront = WFSController.State.FLAT_WAVEFRONT
        self._result = None
        self._optimized_wavefront = None
        self._recompute_wavefront = False
        self._feedback_enhancement = None
        self._test_wavefront = False
        self._snr = None  # Average SNR. Computed when wavefront is computed.
        self._estimated_enhancement = None  # Expected enhancement from phase stepping measurements

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
            self.algorithm._slm.set_phases(0.0)
        else:
            if self._recompute_wavefront or self._optimized_wavefront is None:
                # select only the wavefront and statistics for the first target
                result = self.algorithm.execute().select_target(0)
                self._optimized_wavefront = -np.angle(result.t)
                self._snr = 1.0 / (1.0 / result.noise_factor - 1.0)
                self._estimated_enhancement = result.estimated_enhancement
                self._result = result
            self.algorithm._slm.set_phases(self._optimized_wavefront)

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
    def estimated_enhancement(self) -> float:
        return self._estimated_enhancement

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
            feedback_flat = self.algorithm._feedback.read()
            self.wavefront = WFSController.State.SHAPED_WAVEFRONT
            feedback_shaped = self.algorithm._feedback.read()
            self._feedback_enhancement = float(feedback_shaped.sum() / feedback_flat.sum())

        self._test_wavefront = value
