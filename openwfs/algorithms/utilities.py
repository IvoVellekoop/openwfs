import numpy as np
from types import SimpleNamespace


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
