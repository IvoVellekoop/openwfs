from typing import Union, Sequence, Optional
from .utilities import ExtentType, CoordinateType, unitless

import numpy as np
from astropy.units import Quantity

# shape of a numpy array, or a single integer that is broadcast to a square shape
ShapeType = Union[int, Sequence[int]]

# a scalar quantity with optional unit attached
ScalarType = Union[float, np.ndarray, Quantity]


def tilt(
    x,
    y,
    g,
    phase_offset: float = 0.0,
):
    """Constructs a linear gradient pattern φ=2g·r

    Note, these are the Zernike tilt modes (modes 2 and 3 in the Noll index convention) with normalization
    so that :math:`\\frac{\\int_0^{2\\pi} \\int_0^1 |Z(\\rho, \\phi)|^2 \\rho d\\rho d\\phi}{\\int_0^{2\\pi} \\int_0^1 \\rho d\\rho d\\phi} = 1`.

    Args:
        x: array of the pupil plane coordinates in the x-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        y: array of the pupil plane coordinates in the y-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        g(tuple of two floats): gradient vector.
            When used in the pupil plane of an objective, this patterns causes a displacement of the beam along the x and y dimensions. For x and y in normalised pupil coodinates (i.e. -1 to 1), this patterns causes a displacement of: (gx, gy) * (-2 /π * λ / 2 / numerical_aperturn).
          (Note: a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
        phase_offset: optional additional phase offset to be added to the pattern

    Return:
        An array of the same shape as x and y (or broadcasted), containing the phase values of the tilt pattern.
    """
    return unitless(x * 2 * g[0] + y * 2 * g[1] + phase_offset)


def lens(x, y, f, wavelength, numerical_aperture):
    """Constructs a square texture that represents a wavefront defocus: (f-sqrt(f²+r²)) · 2π/λ

    Args:
        x: array of the normalised pupil plane coordinates in the x-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        y: array of the normalised pupil plane coordinates in the y-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        f: focal length of the lens.
        wavelength: wavelength of the light.
        numerical_aperture: numerical aperture of the lens. This is used to convert the pupil-conjugate coordinates x and y to physical units (i.e. to convert from normalised pupil coodinates to k-space coordinates). An x and y of 1 correspond to the edge of the pupil, convering the full numerical aperture of the lens (applied by the phase mask).

    Return:
        An array of the same shape as x and y (or broadcasted), containing the phase values of the lens pattern.
    """
    k = 2 * np.pi / wavelength
    return unitless(k * f * (1 - np.sqrt(1 + (x**2 + y**2) * numerical_aperture**2)))


def propagation(
    x,
    y,
    distance: ScalarType,
    wavelength: ScalarType,
    refractive_index: ScalarType,
    numerical_aperture: ScalarType,
):
    """
    Computes the phase mask that can be applied in the pupil plane of an objective to digitially propagate the field in the object plane by a distance `distance`.

    Args:
        x: array of the pupil plane coordinates in the x-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        y: array of the pupil plane coordinates in the y-direction. The shape of x and y should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        distance (ScalarType): physical distance to propagate axially.
        refractive_index (Scalar): refractive index of the medium in which the light is propagating.
        wavelength (Scalar): wavelength of the light.
        numerical_aperture: numerical aperture of the lens. This is used to convert the pupil-conjugate coordinates x and y to physical units (i.e. to convert from normalised pupil coodinates to k-space coordinates). An x and y of 1 correspond to the edge of the pupil, convering the full numerical aperture of the lens (applied by the phase mask).
    """
    k = 2 * np.pi * refractive_index / wavelength
    k_x = k * numerical_aperture * x
    k_y = k * numerical_aperture * y
    k_z = np.sqrt(np.maximum(k**2 - k_x**2 - k_y**2, 0))

    return unitless(distance * k_z)


def disk(
    x,
    y,
    radius: ScalarType,
):
    """Constructs an image of a centered (ellipsoid) disk.

    (x)^2 + (y)^2 <= radius^2

    Args:
    """
    return (x**2 + y**2) <= radius**2


def gaussian(
    x,
    y,
    waist: ScalarType,
    truncation_radius: ScalarType = None,
):
    """Constructs an image of a centered Gaussian

    `waist`, `extent` and the optional `truncation_radius` should all have the same unit.

    Args:
        waist (ScalarType): location of the beam waist (1/e value)
            relative to half of the size of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius (ScalarType): when not None, specifies the radius of a disk that is used to truncate the
            Gaussian. All values outside the disk are set to 0.
        extent: see module documentation
        offset: offsets the centre of the Gaussian. The centre of the disk is also offsetted by this amount.

    """
    w2inv = -1.0 / waist**2
    gauss = np.exp(unitless((x**2 + y**2) * w2inv))
    if truncation_radius is not None:
        gauss = gauss * disk(x, y, truncation_radius)
    return unitless(gauss)
