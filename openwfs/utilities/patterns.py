from typing import Union, Sequence, Optional

import numpy as np
from astropy.units import Quantity

from .utilities import ExtentType, CoordinateType, unitless

from . import patterns_f

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


def r2_range(shape: ShapeType, extent: ExtentType, offset: Optional[CoordinateType] = None):
    """Convenience function to return square distance to the origin
    Equivalent to computing cx^2 + cy^2
    """
    c0, c1 = coordinate_range(shape, extent, offset)
    return c0**2 + c1**2


def tilt(
    shape: ShapeType,
    extent: ExtentType,
    g: ExtentType,
    phase_offset: float = 0.0,
    offset: Optional[CoordinateType] = None,
):
    """Constructs a linear gradient pattern φ=2g·r

    Note, these are the Zernike tilt modes (modes 2 and 3 in the Noll index convention) with normalization
    so that :math:`\\frac{\\int_0^{2\\pi} \\int_0^1 |Z(\\rho, \\phi)|^2 \\rho d\\rho d\\phi}{\\int_0^{2\\pi} \\int_0^1 \\rho d\\rho d\\phi} = 1`.

    Args:
        shape: see module documentation
        g(tuple of two floats): gradient vector.
          This has the unit: 1 / extent.unit.
          For the default extent of (2.0, 2.0), a value of g=(1,0)
          corresponds to having a ramp from -2 to +2 over the height of the pattern.
          With an extent of (2.0, 2.0) covering the full NA,
          this pattern causes a displacement of -2/π times the Abbe diffraction limit
          (Note: a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
        extent: see module documentation
        phase_offset: optional additional phase offset to be added to the pattern
    """

    offset = np.multiply(offset, -1) if offset is not None else None
    return unitless(patterns_f.tilt(*coordinate_range(shape, extent), g=g, phase_offset=phase_offset))


def lens(
    shape: ShapeType,
    extent: ExtentType,
    f: ScalarType,
    wavelength: ScalarType,
    numerical_aperture: ScalarType,
    offset=None,
):
    """Constructs a square texture that represents a wavefront defocus: (f-sqrt(f²+r²)) · 2π/λ

    `extent`, `wavelength` and `f` should have compatible units (typically astropy length units).

    Args:
        shape(ShapeType): see module documentation
        f(ScalarType): focal length
        wavelength(ScalarType): wavelength
        extent(ExtentType): physical extent of the SLM, same units as `f` and `wavelength`
    """

    offset = np.multiply(offset, -1) if offset is not None else None

    return patterns_f.lens(
        *coordinate_range(shape, extent, offset=offset),
        f=f,
        wavelength=wavelength,
        numerical_aperture=numerical_aperture,
    )


def propagation(
    shape: ShapeType,
    extent: ExtentType,
    distance: ScalarType,
    wavelength: ScalarType,
    refractive_index: ScalarType,
    numerical_aperture: ScalarType,
    offset=None,
):
    """Computes a wavefront that corresponds to digitally propagating the field in the object plane.

    k_z = sqrt(n² k_0²-k_x²-k_y²)
    φ = k_z · distance

    Args:
          shape: see module documentation
          distance (ScalarType): physical distance to propagate axially.
          refractive_index (Scalar):
          wavelength (Scalar):
            the numerical aperture, refractive index and wavelength are used
            to convert the `extent` from pupil coordinates to k-space (unit radians/meter),
          extent: extent of the returned image,r2_range in NA units. To cover the full NA with a square, use (2*NA, 2*NA)
    """
    offset = np.multiply(offset, -1) if offset is not None else None

    return patterns_f.propagation(
        *coordinate_range(shape, extent, offset=offset),
        distance=distance,
        wavelength=wavelength,
        refractive_index=refractive_index,
        numerical_aperture=numerical_aperture,
    )


def disk(
    shape: ShapeType,
    extent: ExtentType,
    radius: ScalarType,
    offset: Optional[CoordinateType] = None,
):
    """Constructs an image of a centered (ellipsoid) disk.

    (x / rx)^2 + (y / ry)^2 <= 1.0

    Args:
          shape: see module documentation
          radius (ScalarType): radius of the disk, should have the same unit as `extent`.
          extent: see module documentation
          offset: offsets the centre of the disk by offset
    """

    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.disk(*coordinate_range(shape, extent, offset=offset), radius=radius)


def gaussian(
    shape: ShapeType,
    extent: ExtentType,
    waist: ScalarType,
    truncation_radius: ScalarType = None,
    offset: Optional[CoordinateType] = None,
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
        offset: offsets the centre of the Gaussian. The centre of the disk is also offsetted by this amount.

    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.gaussian(
        *coordinate_range(shape, extent, offset=offset), waist=waist, truncation_radius=truncation_radius
    )
