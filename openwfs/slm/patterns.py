import numpy as np
from math import pi

def coordinate_range(resolution):
    """returns a range of the center point coordinates of a texture with the specified resolution and endpoints -1, 1"""
    dx = 2.0 / resolution
    return np.arange(-1.0 + 0.5 * dx, 1.0, dx)


def defocus(resolution):
    """Constructs a square texture that represents a defocus: 2 pi sqrt(1-r^2)
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
    range_sqr = (coordinate_range(resolution) * (2.0 * pi)) ** 2
    r_sqr = ((2 * pi) ** 2 - range_sqr.reshape(resolution, 1)) - range_sqr.reshape(1, resolution)
    return np.sqrt(np.maximum(r_sqr, 0.0)) - pi


def disk(resolution, radius=1.0):
    """Constructs an image of a centered disk. With radius=1.0, the disk touches the sides of the square"""

    # construct coordinate range. The full texture spans the range -1 to 1, and it is divided into N_pixels pixels.
    # The coordinates correspond to the centers of these pixels
    range_sqr = coordinate_range(resolution) ** 2
    r2 = radius ** 2
    return np.sqrt(range_sqr < r2)
