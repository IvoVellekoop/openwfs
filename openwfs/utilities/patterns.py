from typing import Union, Sequence, Optional

import numpy as np
from astropy.units import Quantity

from .utilities import ExtentType, CoordinateType, unitless

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

Optionally, a different `extent` can be specified to scale the coordinates.
This is especially useful when working with anisotropic pixels or rectangular patterns.
For example, a square pattern with anisotropic may be described by shape=(80,100) and extent(2,2)
whereas shape=(80,100) and extent(8,10) describes square pixels that form a rectangle.

In a pupil-conjugate configuration, the transformation matrix of the SLM should be set such that
SLM coordinates correspond to normalized pupil coordinates.
In this case, a disk of extent=(NA, NA) exactly covers the back pupil of the microscope objective.

The extent may have a unit of measure. In this case, other parameters (such as `radius`) may need to have
an according unit of measure.

The (0,0) coordinate is always located in the center of the pattern, which may be on a grid point (for odd shape)
or between grid points (for even shape).

The returned array has a pixel_size property attached.
"""


def coordinate_range(
    shape: ShapeType, extent: ExtentType, offset: Optional[CoordinateType] = None
) -> (Quantity, Quantity):
    """
    Returns coordinate vectors for the two coordinates (y and x).

    Given a range from `-extent/2` to `+extent/2`, uniformly divided into `shape` pixels,
    the returned coordinates correspond to the centers of these coordinates.

    Arguments:
        shape (ShapeType): size of the full grid (y, x) in pixels
        extent (ExtentType): extent of the coordinate range
        offset (Optional[CoordinateType]): offset to be added to the coordinates (optional)

    Returns:
        Tuple[Quantity, Quantity]: coordinate vectors for the two coordinates (y and x)
    """
    if np.size(shape) == 1:
        shape = np.array(shape).repeat(2)

    extent = Quantity(extent)
    if extent.size == 1:
        extent = extent.repeat(2)

    if offset is None:
        offset = 0.0 * extent  # by default, let center be (0,0) (in whatever unit)

    def c_range(res, ex, cx):
        dx = ex / res
        return np.arange(res) * dx + (0.5 * dx - 0.5 * ex + cx)

    return (
        c_range(shape[0], extent[0], offset[0]).reshape((-1, 1)),
        c_range(shape[1], extent[1], offset[1]).reshape((1, -1)),
    )


def r2_range(shape: ShapeType, extent: ExtentType):
    """Convenience function to return square distance to the origin
    Equivalent to computing cx^2 + cy^2
    """
    c0, c1 = coordinate_range(shape, extent)
    return c0**2 + c1**2


def tilt(
    shape: ShapeType,
    g: ExtentType,
    extent: ExtentType = (2.0, 2.0),
    phase_offset: float = 0.0,
):
    """Constructs a linear gradient pattern φ=2 g·r

    Args:
        shape: see module documentation
        g(tuple of two floats): gradient vector.
          This has the unit: 1 / extent.unit.
          For the default extent of (2.0, 2.0), a value of g=(1,0)
          corresponds to having a ramp from -2 to +2 over the height of the pattern
          When this pattern is used as a phase in a pupil-conjugate configuration,
          this corresponds to a displacement of -2/π times the Abbe diffraction limit
          (e.g. a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
        extent: see module documentation
        phase_offset: optional additional phase offset to be added to the pattern
    """
    c0, c1 = coordinate_range(shape, extent * (Quantity(g) * 2.0))
    return unitless(c0 + (c1 + phase_offset))


def lens(shape: ShapeType, f: ScalarType, wavelength: ScalarType, extent: ExtentType):
    """Constructs a square texture that represents a wavefront defocus: (f-sqrt(f²+r²)) · 2π/λ

    `extent`, `wavelength` and `f` should have compatible units (typically astropy length units).

    Args:
        shape(ShapeType): see module documentation
        f(ScalarType): focal length
        wavelength(ScalarType): wavelength
        extent(ExtentType): physical extent of the SLM, same units as `f` and `wavelength`
    """
    r_sqr = r2_range(shape, extent)
    return unitless((f - np.sqrt(f**2 + r_sqr)) * (2 * np.pi / wavelength))


def propagation(
    shape: ShapeType,
    distance: ScalarType,
    numerical_aperture: ScalarType,
    refractive_index: ScalarType,
    wavelength: ScalarType,
    extent: ExtentType = (2.0, 2.0),
):
    """Computes a wavefront that corresponds to digitally propagating the field in the object plane.

    k_z = sqrt(n² k_0²-k_x²-k_y²)
    φ = k_z · distance

    Args:
          shape: see module documentation
          distance (ScalarType): physical distance to propagate axially.
          numerical_aperture (Scalar):
          refractive_index (Scalar):
          wavelength (Scalar):
            the numerical aperture, refractive index and wavelength are used
            to convert the `extent` from pupil coordinates to k-space (unit radians/meter),
          extent: extent of the returned image, in pupil coordinates
            (a disk of radius 1.0 corresponds to the full NA)
    """
    # convert pupil coordinates to absolute k_x, k_y coordinates
    k_0 = 2.0 * np.pi / wavelength
    extent_k = Quantity(extent) * numerical_aperture * k_0
    k_z = np.sqrt(np.maximum((refractive_index * k_0) ** 2 - r2_range(shape, extent_k), 0.0))
    return unitless(k_z * distance)


def disk(shape: ShapeType, radius: ScalarType = 1.0, extent: ExtentType = (2.0, 2.0)):
    """Constructs an image of a centered (ellipsoid) disk.

    (x / rx)^2 + (y / ry)^2 <= 1.0

    Args:
          shape: see module documentation
          radius (ScalarType): radius of the disk, should have the same unit as `extent`.
          extent: see module documentation
    """
    return 1.0 * (r2_range(shape, extent) < radius**2)


def gaussian(
    shape: ShapeType,
    waist: ScalarType,
    truncation_radius: ScalarType = None,
    extent: ExtentType = (2.0, 2.0),
):
    """Constructs an image of a centered Gaussian

    `waist`, `extent` and the optional `truncation_radius` should all have the same unit.

    Args:
        shape: see module documentation
        waist (ScalarType): location of the beam waist (1/e value)
            relative to half of the size of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius (ScalarType): when not None, specifies the radius of a disk that is used to truncate the
            Gaussian. All values outside the disk are set to 0.
        extent: see module documentation

    """
    r_sqr = r2_range(shape, extent)
    w2inv = -1.0 / waist**2
    gauss = np.exp(unitless(r_sqr * w2inv))
    if truncation_radius is not None:
        gauss = gauss * disk(shape, truncation_radius, extent=extent)
    return unitless(gauss)
