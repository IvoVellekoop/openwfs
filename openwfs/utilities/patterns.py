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
For an extent of (2, 2), the returned pattern is assumed to cover a -1,1 x -1,1 square.
In this case, the coordinates range from -1+dx/2 to 1-dx/2, where dx=2.0/shape is the pixel size.

Extent and shape may be specified individually to work with anisotropic pixels or rectangular patterns.
For example, a square pattern with anisotropic pixels may be described by shape=(80,100) and extent(2,2)
whereas shape=(80,100) and extent(8,10) describes square pixels that form a rectangle.

The functions were designed to use normalised pupil coordinates (i.e. an extent of (2,2) which covers the entire back focal plane) so they can be used directly in a pupil-conjugate configuration. 

Some patterns (e.g. disk) depend on the ratio between the extent and other parameters (e.g. radius) so any coordinate system can be used by tuning both parameters. The documentation is written in terms of normalised pupil coordinates for consistency.

The (0,0) coordinate is always located in the center of the pattern, which may be on a grid point (for odd shape)
or between grid points (for even shape).

The returned array has a pixel_size property attached.

Functions are available on openwfs.utilities.patterns_f that accept the coordinates as input instead of requiring an extent and shape.
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
        shape: Number of pixels of the returned pattern.
        extent: extent of the return pattern defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of a microscope objective.
        g(tuple of two floats): gradient vector. For an extent of (2,2), the shift in the focal plane is given by (gx, gy) * -2 / π * wavelength / 2 / numerical_aperture_aperture. Where 'numerical_aperture' is the numerical aperture of the microscope objective, and 'wavelength' is the wavelength of the light.
          (Note: a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
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
    """Constructs a phase mask mimicking a lens: (f-sqrt(f²+r²)) · 2π/λ

    `extent`, `wavelength` and `f` should have compatible units (typically astropy length units).

    Args:
        shape(ShapeType): number of pixels of the returned pattern.
        extent(ExtentType): extent of the return pattern defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of the lens mimicked by the pattern.
        f(ScalarType): focal length
        wavelength(ScalarType): wavelength
        numerical_aperture(ScalarType): numerical aperturn of the lens mimicked by the pattern. This is used to convert the `extent` from normalised pupil coordinates to k-space (unit radians/meter), together with the `wavelength` and `f`.
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
          shape: number of pixels of the returned pattern.
          extent: Extent of the return image. This value is defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of a microscope objective with NA of `numerical_aperture`.
          distance (ScalarType): physical distance to propagate axially.
          refractive_index (Scalar):
          wavelength (Scalar):
            the numerical aperture, refractive index and wavelength are used
            to convert the `extent` from pupil coordinates to k-space (unit radians/meter),
          numerical_aperture: numerical aperture of the microscope objective. This is used to convert the `extent` from pupil coordinates to k-space, together with the `wavelength` and `refractive_index`.

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
          shape: number of pixels of the returned pattern.
          extent: extent of the return pattern. This value is used to compute the coordinates of each pixel of the image. Scaling both the extent and radius by the same factor does not change the returned pattern, but changing their ratio does.
          radius (ScalarType): radius of the disk, should have the same unit as `extent`.
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
        shape: Number of pixels of the returned pattern.
        extent: Extent of the return pattern. This value is used to compute the coordinates for the Gaussian profile. Changing the ratio of `extent` and `waist` changes the returned pattern, but scaling both by the same factor does not change the returned pattern.
        waist (ScalarType): location of the beam waist (1/e value)
            relative to half of the size of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius (ScalarType): when not None, specifies the radius of a disk that is used to truncate the
            Gaussian. All values outside the disk are set to 0.
        offset: offsets the centre of the Gaussian. The centre of the disk is also offsetted by this amount.

    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.gaussian(
        *coordinate_range(shape, extent, offset=offset), waist=waist, truncation_radius=truncation_radius
    )
