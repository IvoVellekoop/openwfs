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

def get_dense_matrix(k_set, t_set):
    """
    Create a dense matrix visualization for given x, y coordinates and complex data.
    Grid points not in the provided data are set to NaN.

    Parameters:
    - k_set: Arrays of x and y coordinates.
    - t_set: Array of complex data corresponding to x, y coordinates.

    Returns:
    - dense_matrix: A 2D numpy array with data filled in the corresponding x, y locations and NaN elsewhere.
    """
    x = k_set[0, :]
    y = k_set[1, :]

    # Find the unique sorted coordinates
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    # Create a mapping from coordinate to index
    x_to_index = {x_val: idx for idx, x_val in enumerate(unique_x)}
    y_to_index = {y_val: idx for idx, y_val in enumerate(unique_y)}

    # Initialize a dense matrix full of NaNs
    dense_matrix = np.full((len(unique_y), len(unique_x)), np.nan, dtype=complex)

    # Place the data into the dense matrix using the mapping
    for (x_val, y_val), val in zip(zip(x, y), t_set):
        x_idx = x_to_index[x_val]
        y_idx = y_to_index[y_val]
        dense_matrix[y_idx, x_idx] = val

    return dense_matrix