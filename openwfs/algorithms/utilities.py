import numpy as np
from enum import Enum


class WFSResult:
    def __init__(self, t, snr=None, noise_factor=None, amplitude_factor=None, estimated_improvement=None):
        self.t = t
        self.snr = snr
        self.noise_factor = noise_factor
        self.amplitude_factor = amplitude_factor
        self.estimated_improvement = estimated_improvement

    def select_target(self, b) -> 'WFSResult':
        return WFSResult(t=self.t.reshape((*self.t.shape[0:2], -1))[:, :, b],
                         snr=None if self.snr is None else self.snr[:][b],
                         noise_factor=None if self.noise_factor is None else self.noise_factor[:][b],
                         amplitude_factor=None if self.amplitude_factor is None else self.amplitude_factor[:][b],
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
    noise_factor = np.maximum(signal_energy - noise_energy, 0.0) / (signal_energy + noise_energy)

    # compute the effect of amplitude variations.
    # for perfectly developed speckle, and homogeneous illumination, this factor will be pi/4
    amplitude_factor = np.mean(np.abs(t), axis=dims) ** 2 / np.mean(np.abs(t) ** 2, axis=dims)

    return WFSResult(t, snr, noise_factor, amplitude_factor=amplitude_factor,
                     estimated_improvement=estimated_improvement)


class WFSController:
    class State(Enum):
        FLAT_WAVEFRONT = 0
        SHAPED_WAVEFRONT = 1

    def __init__(self, algorithm):
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
        return self._wavefront

    @wavefront.setter
    def wavefront(self, value):
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
        if value:
            self.wavefront = WFSController.State.FLAT_WAVEFRONT
            feedback_flat = self._algorithm._feedback.read()
            self.wavefront = WFSController.State.SHAPED_WAVEFRONT
            feedback_shaped = self._algorithm._feedback.read()
            self._feedback_enhancement = float(feedback_shaped.sum() / feedback_flat.sum())

        self._test_wavefront = value
