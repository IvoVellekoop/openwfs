import numpy as np
from typing import Union, Sequence


# Each of the functions in this module computes a square pattern with a given resolution for x and y dimensions
# The coordinate system that is used assumes that the pixels in the pattern fill a range from -1.0 to 1.0.
# All computations are then done on the coordinates that represent the  _centers_ of these pixels.

def coordinate_range(resolution: Union[int, Sequence[int]]):
    """Returns a coordinate vectors for the two coordinates (y and x)

    If resolution is a scalar, assumes the same resolution for both axes.
    """
    if np.size(resolution) == 1:
        resolution = (resolution, resolution)

    dy = 2.0 / resolution[0]
    dx = 2.0 / resolution[1]
    return (np.arange(-1.0 + 0.5 * dy, 1.0, dy).reshape((-1, 1)),
            np.arange(-1.0 + 0.5 * dx, 1.0, dx).reshape((1, -1)))


def r2_range(resolution: Union[int, Sequence[int]]):
    """Convenience function to return square distance to the origin
    Equivalent to computing cx^2 + cy^2
    """
    c0, c1 = coordinate_range(resolution)
    return c0 ** 2 + c1 ** 2


def tilt(resolution: Union[int, Sequence[int]], slope):
    """
    Args:
        resolution:
        slope(tuple of two doubles): number of 2π phase wraps in both directions (y,x)

    Returns:

    """
    slope = np.array(slope)
    c0, c1 = coordinate_range(resolution)
    slope_2pi = np.pi * slope
    return slope_2pi[1] * c1 + slope_2pi[0] * c0


def defocus(resolution: Union[int, Sequence[int]]):
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
    r_sqr = r2_range(resolution)
    return (2 * np.pi) * np.sqrt(np.maximum(1.0 - r_sqr, 0.0)) - np.pi


def disk(resolution: Union[int, Sequence[int]], radius=1.0):
    """Constructs an image of a centered disk. With radius=1.0, the disk touches the sides of the square"""
    return 1.0 * (r2_range(resolution) < radius ** 2)


def gaussian(resolution: Union[int, Sequence[int]], waist, truncation_radius=None):
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
    r_sqr = r2_range(resolution)
    w2inv = -1.0 / waist ** 2
    gauss = np.exp(r_sqr * w2inv)
    if truncation_radius is not None:
        gauss = gauss * disk(resolution, truncation_radius)
    return gauss
