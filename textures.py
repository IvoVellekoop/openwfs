import numpy as np
from math import pi


def defocus(N_pixels):
    """Constructs a texture that represents a defocus: 2 pi sqrt(1-r^2)
    The wavefront ranges from 0 (min) to 2pi (max)
    For a pupil-conjugate configuration, with the wavefront displayed on a 1.0 'radius' square, this texture causes
    the focus to shift to one wavelength further away from the lens (deeper).
    Justification:
    Coordinates correspond to sin(theta). This function then computes 2 pi cos(theta), which corresponds to
    k_z lambda phase shift, i.e. one wavelength displacement in the z-direction.
    the defocus can be computed as f = lambda / ...
    """

    # construct coordinate range. The full texture spans the range -1 to 1, and it is divided into N_pixels pixels.
    # The coordinates correspond to the centers of these pixels
    dx = 2.0 / N_pixels
    range_sqr = (np.arange(-1.0 + 0.5 * dx, 1.0, dx) * (2.0 * pi)) ** 2
    r_sqr = ((2 * pi) ** 2 - range_sqr.reshape(N_pixels, 1)) - range_sqr.reshape(1, N_pixels)
    return np.sqrt(np.maximum(r_sqr, 0.0)) - pi
