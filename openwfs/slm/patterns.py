import numpy as np
from typing import Union, Sequence
import astropy.units as u
from astropy.units import Quantity

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

ShapeType = Union[int, Sequence[int]]
ExtentType = Union[Sequence[float], np.ndarray, Quantity]
ScalarType = Union[float, np.ndarray, Quantity]


def coordinate_range(shape: ShapeType, extent: ExtentType):
    """Returns coordinate vectors for the two coordinates (y and x)"""
    if isinstance(shape, Quantity):
        shape = shape.to_value(u.dimensionless_unscaled)

    if np.size(shape) == 1:
        shape = (shape, shape)

    def c_range(res, ex):
        dx = ex / res
        return np.arange(res) * dx + (0.5 * dx - 0.5 * ex)

    return (c_range(shape[0], extent[0]).reshape((-1, 1)),
            c_range(shape[1], extent[1]).reshape((1, -1)))


def r2_range(shape: ShapeType, extent: ExtentType):
    """Convenience function to return square distance to the origin
    Equivalent to computing cx^2 + cy^2
    """
    c0, c1 = coordinate_range(shape, extent)
    return c0 ** 2 + c1 ** 2


def tilt(shape: ShapeType, k: ExtentType, extent: ExtentType = (2.0, 2.0)):
    """Constructs a linear gradient pattern

    Args:
        shape: see module documentation
        k(tuple of two floats): perpendicular wave vector.
          This has the unit: radians / extent.unit.
          For the default extent of (2.0, 2.0), a value of k=(π,0)
          corresponds to having a 2π phase ramp over the height of the pattern (from -π to +π)
          When this pattern is used as a phase, this corresponds to a periodicity of 1.
        extent: see module documentation
    """
    k = Quantity(k)
    c0, c1 = coordinate_range(shape, extent * k)
    return c0 + c1


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
    return Quantity((f - np.sqrt(np.maximum(f ** 2 - r_sqr, 0.0))) * (2 * np.pi / wavelength)).to_value(
        u.dimensionless_unscaled)


def disk(shape: ShapeType, radius: ScalarType = 1.0, extent: ExtentType = (2.0, 2.0)):
    """Constructs an image of a centered (ellipsoid) disk.

    (x / rx)^2 + (y / ry)^2 <= 1.0

    Args:
          shape: see module documentation
          radius (ScalarType): radius of the disk, should have the same unit as `extent`.
          extent: see module documentation
    """
    return 1.0 * (r2_range(shape, extent) < radius ** 2)


def gaussian(shape: ShapeType, waist: ScalarType,
             truncation_radius: ScalarType = None, extent: ExtentType = (2.0, 2.0)):
    """Constructs an image of a centered Gaussian

    `waist`, `extent` and the optional `truncation_radius` should all have the same unit.
    Arguments:
        shape: see module documentation
        waist (ScalarType):
            location of the beam waist (1/e value)
            relative to half of the size of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius (ScalarType):
            when not None, specifies the radius of a disk that is used to truncate the Gaussian.
            All values outside the disk are set to 0.
        extent: see module documentation
    """
    r_sqr = r2_range(shape, extent)
    w2inv = -1.0 / waist ** 2
    gauss = np.exp(r_sqr * w2inv)
    if truncation_radius is not None:
        gauss = gauss * disk(shape, truncation_radius, extent=extent)
    return gauss
