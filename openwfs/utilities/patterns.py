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
    g: ExtentType,
    extent: ExtentType = (2.0, 2.0),
    phase_offset: float = 0.0,
    offset: Optional[CoordinateType] = None,
):
    """Constructs a linear gradient pattern φ=2g·r

    Note, these are the Zernike tilt modes (modes 2 and 3 in the Noll index convention) with normalization
    so that :math:`\\frac{\\int_0^{2\\pi} \\int_0^1 |Z(\\rho, \\phi)|^2 \\rho d\\rho d\\phi}{\\int_0^{2\\pi} \\int_0^1 \\rho d\\rho d\\phi} = 1`.

    Args:
        shape: Number of pixels of the returned pattern.
        extent: extent of the return pattern defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of a microscope objective.
        g(tuple of two floats): gradient vector. For an extent of (2,2), the shift in the focal plane is given by (gx, gy) * -1 / π * wavelength / numerical_aperture_aperture. Where 'numerical_aperture' is the numerical aperture of the microscope objective, and 'wavelength' is the wavelength of the light.
          (Note: a positive x-gradient g causes the focal point to move in the _negative_ x-direction)
        phase_offset: optional additional phase offset to be added to the pattern
    """

    offset = np.multiply(offset, -1) if offset is not None else None
    g = Quantity(g)
    if g.size == 1:
        g = g.repeat(2)
    return unitless(patterns_f.tilt(*coordinate_range(shape, extent), gx=g[0], gy=g[1], phase_offset=phase_offset))


def lens(
    shape: ShapeType,
    f: Quantity,
    wavelength: Quantity,
    numerical_aperture: float,
    extent: ExtentType = (2.0, 2.0),
    offset=None,
):
    """Constructs a phase mask mimicking a lens: (f-sqrt(f²+r²)) · 2π/λ

    `extent`, `wavelength` and `f` should have compatible units (typically astropy length units).

    Args:
        shape: number of pixels of the returned pattern.
        extent: extent of the return pattern defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of the lens mimicked by the pattern.
        f: focal length
        wavelength: wavelength
        numerical_aperture: numerical aperturn of the lens mimicked by the pattern. This is used to convert the `extent` from normalised pupil coordinates to k-space (unit radians/meter), together with the `wavelength` and `f`.

    Returns:
        An array of the same shape as the input `shape`, containing the phase values of the lens phase mask.
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
    distance: Quantity,
    wavelength: Quantity,
    numerical_aperture: float,
    refractive_index: float = 1.0,
    extent: ExtentType = (2.0, 2.0),
    offset=None,
):
    """Computes a wavefront that corresponds to digitally propagating the field in the object plane.

    k_z = sqrt(n² k_0²-k_x²-k_y²)
    φ = k_z · distance

    Args:
          shape: number of pixels of the returned pattern.
          extent: Extent of the return image. This value is defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of a microscope objective with NA of `numerical_aperture`.
          distance (Quantity): physical distance to propagate axially.
          wavelength (Quantity): wavelength of the light.
          refractive_index (float): refractive index of the medium in which the light is propagating.
          numerical_aperture (float): numerical aperture of the microscope objective. This is used to convert the `extent` from pupil coordinates to k-space, together with the `wavelength` and `refractive_index`.

    Return
            An array of the same shape as the input `shape`, containing the phase values of the wavefront.

    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.propagation(
        *coordinate_range(shape, extent, offset=offset),
        distance=distance,
        wavelength=wavelength,
        refractive_index=refractive_index,
        numerical_aperture=numerical_aperture,
    )


def parabola(
    shape: ShapeType,
    alpha: ScalarType,
    extent: ExtentType = (2.0, 2.0),
    offset: Optional[CoordinateType] = None,
):
    """Constructs a parabola phase mask: alpha * (x^2 + y^2)

    `extent` and `alpha` should have compatible units (typically astropy length units).

    Args:
          shape: number of pixels of the returned pattern.
          extent: Extent of the return image. This value is defined in normalised pupil coordinates, i.e. an extent of (2, 2) covers the entire back pupil plane of a microscope objective.
          alpha (float): coefficient of the parabola phase mask. This is used together with the `extent` to determine the curvature of the parabola.
          offset: offsets the centre of the parabola by offset. If the parabola is not centered on the back pupil plane, the image in the focal plane will be shifted. The resulting shift is given by offset * alpha * wavelength / (numerical_aperture * π), where `numerical_aperture` is the numerical aperture of the microscope objective, and `wavelength` is the wavelength of the light.

    Return:
            An array of the same shape as the input `shape`, containing the phase values of the parabola phase mask.

    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.parabola(*coordinate_range(shape, extent, offset=offset), alpha=alpha)


def disk(
    shape: ShapeType,
    radius: float = 1,
    extent: ExtentType = (2.0, 2.0),
    offset: Optional[CoordinateType] = None,
):
    """Constructs an image of a centered (ellipsoid) disk.

    (x / rx)^2 + (y / ry)^2 <= 1.0

    Args:
          shape: number of pixels of the returned pattern.
          extent: extent of the return pattern. This value is used to compute the coordinates of each pixel of the image. Scaling both the extent and radius by the same factor does not change the returned pattern, but changing their ratio does.
          radius (ScalarType): radius of the disk, should have the same unit as `extent`.
          offset: offsets the centre of the disk by offset

    Return:
            An array of the same shape as the input `shape`, containing the values of the disk pattern. The values are 1 inside the disk and 0 outside the disk.
    """

    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.disk(*coordinate_range(shape, extent, offset=offset), radius=radius)


def gaussian(
    shape: ShapeType,
    waist: ScalarType,
    extent: ExtentType = (2.0, 2.0),
    truncation_radius: ScalarType = None,
    offset: Optional[CoordinateType] = None,
):
    """Constructs an image of a centered Gaussian

    `waist`, `extent` and the optional `truncation_radius` should all have the same unit.

    Args:
        shape: Number of pixels of the returned pattern.
        extent: Extent of the return pattern. This value is used to compute the coordinates for the Gaussian profile. Changing the ratio of `extent` and `waist` changes the returned pattern, but scaling both by the same factor does not change the returned pattern.
        waist: location of the beam waist (1/e value)
            relative to half of the size of the pattern (i.e. relative to the `radius` of the square)
        truncation_radius: when not None, specifies the radius of a disk that is used to truncate the
            Gaussian. All values outside the disk are set to 0.
        offset: offsets the centre of the Gaussian. The centre of the disk is also offsetted by this amount.

    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.gaussian(
        *coordinate_range(shape, extent, offset=offset), waist=waist, truncation_radius=truncation_radius
    )


def binary_grating(
    shape: ShapeType,
    period: ScalarType,
    values: Sequence[float],
    extent: ExtentType = (2.0, 2.0),
    angle: ScalarType = 0.0,
    offset: Optional[CoordinateType] = None,
):
    """Constructs a binary grating pattern.

    Args:
        shape: Number of pixels of the returned pattern.
        extent: Extent of the return pattern. This value is used to compute the coordinates for the grating pattern. Changing the ratio of `extent` and `period` changes the returned pattern, but scaling both by the same factor does not change the returned pattern.
        period (ScalarType): period of the grating, should have the same unit as `extent`.
        values: tuple of two values (v0, v1) that are used for the two levels of the binary grating. For example, for a binary phase grating, these values could be (0, π).
        angle (ScalarType): angle of the grating in radians. For an angle of 0, the grating is oriented along the x-axis, and for an angle of π/2, the grating is oriented along the y-axis.
        offset: offsets the centre of the grating by offset

        For a SLM in a pupil-conjugate configuration with an objective: If the extent and periodicity is defined in the normalised pupil coordinates, the image created by the first diffraction order of the grating is shifted in the focal plane by wavelength / period / na * (cos(angle), sin(angle)).

    Returns:
        An array of the same shape as the input `shape`, containing the phase values of the binary grating pattern.
    """
    offset = np.multiply(offset, -1) if offset is not None else None
    return patterns_f.binary_grating(
        *coordinate_range(shape, extent, offset=offset), period=period, values=values, angle=angle
    )
