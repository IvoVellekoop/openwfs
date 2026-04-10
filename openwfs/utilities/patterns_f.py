from typing import Union, Sequence, Optional
from .utilities import ExtentType, CoordinateType, unitless

import numpy as np
from astropy.units import Quantity

# shape of a numpy array, or a single integer that is broadcast to a square shape
ShapeType = Union[int, Sequence[int]]

# a scalar quantity with optional unit attached
ScalarType = Union[float, np.ndarray, Quantity]

"""
Library of functions to create commonly used patterns

Each of the functions takes a `shape` input, which may be a scalar integer or a 2-element Sequence of integers
indicating the size (shape) in pixels of the returned field. If `shape` is a scalar, the same value is
used for both axes.

For the coordinates, the OpenGL convention is used, where the coordinates indicate the centers of the pixels.
By default, the returned pattern is assumed to cover a -1,1 x -1,1 square, 
which corresponds to a default `extent` parameter of (2.0, 2.0).
In this case, the coordinates range from -1+dx/2 to 1-dx/2, where dx=2.0/shape is the pixel size.

Exptent and shape may be specified individually to work with anisotropic pixels or rectangular patterns.
For example, a square pattern with anisotropic pixels may be described by shape=(80,100) and extent(2,2)
whereas shape=(80,100) and extent(8,10) describes square pixels that form a rectangle.

In a pupil-conjugate configuration, a disk of extent=(NA, NA) exactly covers the back pupil of the microscope objective.
The transformation matrix of the SLM should be set such that SLM coordinates correspond to normalized pupil coordinates.

The extent may have a unit of measure. In this case, other parameters (such as `radius`) may need to have
an according unit of measure.

The (0,0) coordinate is always located in the center of the pattern, which may be on a grid point (for odd shape)
or between grid points (for even shape).

The returned array has a pixel_size property attached.
"""


def tilt(
    rx,
    ry,
    g,
    phase_offset: float = 0.0,
):
    """Constructs a linear gradient pattern φ=2g·r

    Note, these are the Zernike tilt modes (modes 2 and 3 in the Noll index convention) with normalization
    so that :math:`\\frac{\\int_0^{2\\pi} \\int_0^1 |Z(\\rho, \\phi)|^2 \\rho d\\rho d\\phi}{\\int_0^{2\\pi} \\int_0^1 \\rho d\\rho d\\phi} = 1`.

    Args:
        rx: array of the pupil plane coordinates in the x-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        ry: array of the pupil plane coordinates in the y-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        g(tuple of two floats): gradient vector.
            When used in the pupil plane of an objective, this patterns causes a displacement of the beam along the x and y dimensions. For rx and ry in pupil coordinates (i.e. -1 to 1), this patterns causes a displacement of: (gx, gy) * (-2 /π * λ / 2 / numerical_aperturn).
          (Note: a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
        phase_offset: optional additional phase offset to be added to the pattern

    Return:
        An array of the same shape as rx and ry (or broadcasted), containing the phase values of the tilt pattern.
    """
    return unitless(rx * 2 * g[0] + ry * 2 * g[1] + phase_offset)


def lens(rx, ry, f, wavelength, numerical_aperture):
    """Constructs a square texture that represents a wavefront defocus: (f-sqrt(f²+r²)) · 2π/λ

    Args:
        rx: array of the pupil plane coordinates in the x-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        ry: array of the pupil plane coordinates in the y-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        f: focal length of the lens.
        wavelength: wavelength of the light.
        numerical_aperture: numerical aperture of the lens. This is used to convert the pupil-conjugate coordinates rx and ry to physical units (i.e. to convert from pupil coordinates to k-space coordinates). An rx and ry of 1 correspond to the edge of the pupil, convering the full numerical aperture of the lens (applied by the phase mask).

    Return:
        An array of the same shape as rx and ry (or broadcasted), containing the phase values of the lens pattern.
    """
    k = 2 * np.pi / wavelength
    return unitless(k * f * (1 - np.sqrt(1 + (rx**2 + ry**2) * numerical_aperture**2)))


def propagation(
    rx,
    ry,
    distance: ScalarType,
    wavelength: ScalarType,
    refractive_index: ScalarType,
    numerical_aperture: ScalarType,
):
    """
    Computes the phase mask that can be applied in the pupil plane of an objective to digitially propagate the field in the object plane by a distance `distance`.

    Args:
        rx: array of the pupil plane coordinates in the x-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        ry: array of the pupil plane coordinates in the y-direction. The shape of rx and ry should be such that they can be added together (i.e. they should be the same shape, or one of them should be broadcastable to the shape of the other).
        distance (ScalarType): physical distance to propagate axially.
        refractive_index (Scalar): refractive index of the medium in which the light is propagating.
        wavelength (Scalar): wavelength of the light.
        numerical_aperture: numerical aperture of the lens. This is used to convert the pupil-conjugate coordinates rx and ry to physical units (i.e. to convert from pupil coordinates to k-space coordinates). An rx and ry of 1 correspond to the edge of the pupil, convering the full numerical aperture of the lens (applied by the phase mask).
    """
    k = 2 * np.pi * refractive_index / wavelength
    k_x = k * numerical_aperture * rx
    k_y = k * numerical_aperture * ry
    k_z = np.sqrt(k**2 - k_x**2 - k_y**2)

    return unitless(distance * k_z)


def disk(
    rx,
    ry,
    radius: ScalarType,
):
    """Constructs an image of a centered (ellipsoid) disk.

    (rx)^2 + (ry)^2 <= radius^2

    Args:
    """
    return (rx**2 + ry**2) <= radius**2


def gaussian(
    rx,
    ry,
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
    gauss = np.exp(unitless((rx**2 + ry**2) * w2inv))
    if truncation_radius is not None:
        gauss = gauss * disk(rx, ry, truncation_radius)
    return unitless(gauss)
