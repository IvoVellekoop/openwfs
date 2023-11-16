import numpy as np
from math import pi


# Each of the functions in this module computes a square pattern with a given resolution for x and y dimensions
# The coordinate system that is used assumes that the pixels in the pattern fill a range from -1.0 to 1.0.
# All computations are then done on the coordinates that represent the  _centers_ of these pixels.

def coordinate_range(resolution):
    """returns a column vector containing the center point coordinates of a texture with endpoints -1, 1"""
    dx = 2.0 / resolution
    return np.arange(-1.0 + 0.5 * dx, 1.0, dx).reshape((-1, 1))


def tilt(resolution, slope):
    """
    Args:
        resolution:
        slope(tuple of two doubles): number of 2π phase wraps in both directions

    Returns:

    """
    slope = np.array(slope)
    coordinates = coordinate_range(resolution)
    slope_2pi = np.pi * slope
    return slope_2pi[1] * coordinates + slope_2pi[0] * coordinates.T


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
    range_sqr = coordinate_range(resolution) ** 2
    r2 = radius ** 2
    return 1.0 * ((range_sqr + range_sqr.T) < r2)


def gaussian(resolution, waist, truncation_radius=None):
    """Constructs an image of a centered gaussian
    Arguments:
        resolution (int):
            width and height (in pixels) of the returned pattern.
        waist (float):
            location of the beam waist (1/e value)
            relative to half of the width of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius (float or None):
            when not None, specifies the radius of a disk that is used to truncate the Gaussian.
            All values outside the disk are set to 0.
    """
    range_sqr = coordinate_range(resolution) ** 2
    w2inv = -1.0 / waist ** 2
    gauss = np.exp((range_sqr + range_sqr.T) * w2inv)
    if truncation_radius is not None:
        gauss = gauss * disk(truncation_radius)
    return gauss
