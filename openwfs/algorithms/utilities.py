import numpy as np
from types import SimpleNamespace
from enum import Enum


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

    With `P` phase steps, the measurements are given by
    .. math::
        I_p = \lvert A + B exp(i 2\pi p / P)\rvert^2,

    This function computes the Fourier transform. math::
        \frac{1}{P} \sum I_p exp(-i 2\pi p / P) = A^* B

    The value of A^* B for each set of measurements is stored in the `field` attribute of the return
    value.
    Other attributes hold an estimate of the signal-to-noise ratio,
    and an estimate of the maximum enhancement that can be expected
    if these measurements are used for wavefront shaping.
    """
    P = measurements.shape[axis]
    phases = np.arange(P) * 2.0 * np.pi / P
    AB = np.tensordot(measurements, np.exp(-1.0j * phases) / P, ((axis,), (0,)))
    return SimpleNamespace(field=AB)


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
                t = self._algorithm.execute()
                t = t.reshape((*t.shape[0:2], -1))
                self._optimized_wavefront = -np.angle(t[:, :, 0])
            self._slm.set_phases(self._optimized_wavefront)

    @property
    def recompute_wavefront(self) -> bool:
        return self._recompute_wavefront

    @recompute_wavefront.setter
    def recompute_wavefront(self, value):
        self._recompute_wavefront = value
