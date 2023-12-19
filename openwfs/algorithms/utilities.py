import numpy as np
from enum import Enum


class WFSResult:
    """
    Data structure for holding wavefront shaping results and statistics.

    Attributes:
        t: measured transmission matrix.
            if multiple targets were used, the first dimension(s) of t denote the columns of the transmission matrix (`a` indices)
            and the last dimensions(s) denote the targets, i.e., the rows of the transmission matrix (`b` indices).
        n: number of degrees of freedom (columns of the transmission matrix)
        snr: estimated signal-to-noise ratio for each of the targets.
        noise_factor: the estimated loss in fidelity caused by the the limited snr.
        amplitude_factor: estimated reduction of the fidelity due to phase-only modulation (≈ π/4 for fully developed speckle)
        estimated_improvement: estimated ratio after/before
        estimated_enhancement: estimated ratio <after>/<before>  (with <> denoting ensemble average)
    """

    def __init__(self, t, snr, amplitude_factor, estimated_improvement, n=None):
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
        self.n = t.size / snr.size if n is None else n
        self.snr = np.atleast_1d(snr)
        self.noise_factor = np.atleast_1d(snr / (snr + 1.0))
        self.amplitude_factor = np.atleast_1d(amplitude_factor)
        self.estimated_improvement = np.atleast_1d(estimated_improvement)
        self.estimated_enhancement = np.atleast_1d(1.0 + (self.n - 1) * self.amplitude_factor * self.noise_factor)

    def select_target(self, b) -> 'WFSResult':
        """
        Returns the wavefront shaping results for a single target

        Args:
            b(int): target to select, as integer index.
                If the target array is multi-dimensional, it is flattened before selecting the `b`-th component.

        Returns: WFSResults data for the specified target
        """
        return WFSResult(t=self.t.reshape((*self.t.shape[0:2], -1))[:, :, b],
                         snr=self.snr[:][b],
                         amplitude_factor=self.amplitude_factor[:][b],
                         estimated_improvement=self.estimated_improvement[:][b]
                         )


def analyze_phase_stepping(measurements: np.ndarray, axis: int):
    """Takes phase stepping measurements and reconstructs the relative field.

    This function assumes that there were two fields interfering at the detector,
    and that the phase of one of these fields is phase-stepped in equally spaced steps
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
        I_p = \lvert A + B exp(i 2\pi p / phase_steps)\rvert^2,

    This function computes the Fourier transform. math::
        \frac{1}{phase_steps} \sum I_p exp(-i 2\pi p / phase_steps) = A^* B

    The value of A^* B for each set of measurements is stored in the `field` attribute of the return
    value.
    Other attributes hold an estimate of the signal-to-noise ratio,
    and an estimate of the maximum enhancement that can be expected
    if these measurements are used for wavefront shaping.
    """
    phase_steps = measurements.shape[axis]
    dims = tuple(range(axis))
    n = np.sum(measurements.shape[:axis])

    # put the phase axis at 0, then do a Fourier transform
    t_f = np.fft.fft(np.moveaxis(measurements, axis, 0), axis=0)

    # the signal is the first Fourier component (lowest non-zero frequency)
    t = t_f[1, ...]
    signal_energy = np.sum(np.abs(t) ** 2, axis=dims)
    offset_energy = np.sum(np.abs(t_f[0, ...]) ** 2, axis=dims)
    total_energy = np.sum(np.abs(t_f) ** 2, axis=(*dims, axis))
    if phase_steps > 3:
        noise_energy = (total_energy - 2.0 * signal_energy - offset_energy) / (phase_steps - 3)
    else:
        noise_energy = 0.0  # cannot estimate reliably

    # estimate the signal improvement that we expect (needs at least four phase steps)
    # t_m = measured t
    # ξ = measurement error
    # Σ = sum over 'a'
    # before:
    # <|Σ(t_m - ξ) · 1|²> = |Σ t_m|² + <|Σ ξ|²> = |Σ t_m|² + Σ<|ξ|²>
    #
    # after:
    # <|Σ(t_m - ξ) t_m^* / |t_m||²> = (Σ|t_m|)² + <|Σ ξ t_m^*/|t_m||²> = (Σ|t_m|)² + Σ<|ξ|²>
    #
    # compute the total energy in the signal, the offset, and the total measurement set. Sum over all segments

    snr = np.maximum(signal_energy - noise_energy, 0.0) / noise_energy

    #    before = noise_energy + np.abs(np.sum(t, axis=dims)) ** 2
    # a2_plus_b2 = np.mean(np.moveaxis(measurements, axis, 0)[0, ...], axis=dims) * (phase_steps ** 2)
    # a2_times_b2 = signal_energy
    # a2_est = 0.5 * (a2_plus_b2 + np.sqrt(a2_plus_b2 ** 2 - 4 * a2_times_b2))
    # b2_est = a2_plus_b2 - a2_est
    a2_plus_b2 = np.mean(np.moveaxis(measurements, axis, 0)[0, ...], axis=dims)
    a2_est = a2_plus_b2 / (1.0 + 1.0 / n)
    t = t / np.sqrt(a2_est) / phase_steps
    before = a2_est
    after = noise_energy + np.sum(np.abs(t), axis=dims) ** 2
    estimated_improvement = after / before

    # compute the effect of amplitude variations.
    # for perfectly developed speckle, and homogeneous illumination, this factor will be pi/4
    amplitude_factor = np.mean(np.abs(t), axis=dims) ** 2 / np.mean(np.abs(t) ** 2, axis=dims)

    return WFSResult(t, snr, amplitude_factor=amplitude_factor, estimated_improvement=estimated_improvement)


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
        """
        self._algorithm = algorithm
        self._slm = algorithm._slm
        self._wavefront = WFSController.State.FLAT_WAVEFRONT
        self._optimized_wavefront = None
        self._recompute_wavefront = False
        self._feedback_enhancement = None
        self._test_wavefront = False
        self._snr = None  # Average SNR. Computed when wavefront is computed.

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
            self._slm.set_phases(0.0)
        else:
            if self._recompute_wavefront or self._optimized_wavefront is None:
                # select only the wavefront and statistics for the first target
                result = self._algorithm.execute().select_target(0)
                self._optimized_wavefront = -np.angle(result.t)
                self._snr = result.snr
            self._slm.set_phases(self._optimized_wavefront)

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
        return self._recompute_wavefront

    @recompute_wavefront.setter
    def recompute_wavefront(self, value):
        self._recompute_wavefront = value

    @property
    def feedback_enhancement(self) -> float:
        return self._feedback_enhancement

    @property
    def test_wavefront(self) -> bool:
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
            feedback_flat = self._algorithm._feedback.read()
            self.wavefront = WFSController.State.SHAPED_WAVEFRONT
            feedback_shaped = self._algorithm._feedback.read()
            self._feedback_enhancement = float(feedback_shaped.sum() / feedback_flat.sum())

        self._test_wavefront = value
