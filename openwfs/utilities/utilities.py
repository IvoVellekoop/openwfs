from dataclasses import dataclass
from typing import Union, Sequence, Optional

import cv2
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import ArrayLike

# A coordinate is a sequence of two floats with an optional unit attached
CoordinateType = Union[Sequence[float], np.ndarray, Quantity]

# A transform is a 2x2 array with optional units attached
TransformType = Union[np.ndarray, Quantity, Sequence[Sequence[float]]]

# An extent is like a coordinate type.
# If it has a single element, this element is automatically reproduced along all dimensions
ExtentType = Union[float, CoordinateType]


def unitless(data: ArrayLike) -> np.ndarray:
    """
    Converts an object to a numpy array.
    Especially useful to remove the scaling from a unitless Quantity.

    Args:
        data: The input data.
        If `data` is a Quantity, it is converted to a (unitless) numpy array.
        All other data types are just returned as is.

    Returns:
        ArrayLike: unitless numpy array, or the input data if it is not a Quantity.

    Raises:
        UnitConversionError: If the data is a Quantity with a unit

    Note:
        Do NOT use `np.array(data)` to convert a Quantity to a numpy array,
        because this will drop the unit prefix.
        For example, `np.array(1 * u.s / u.ms) == 1`.
        Whereas `unitless(1 * u.s / u.ms) == 1000` gives the correct answer.

    Usage:
    >>> data = np.array([1.0, 2.0, 3.0]) * u.m
    >>> unitless_data = unitless(data / u.mm)
    """
    if isinstance(data, Quantity):
        return data.to_value(u.dimensionless_unscaled)
    else:
        return np.array(data)


@dataclass
class Transform:
    """Represents a transformation from one 2-d coordinate system to another.

    Transform objects are used to specify any combination of shift, (anisotropic) scaling, rotation and shear.
    The transform matrix, as well as the source and destination origins, can be specified with astropy units attached.

    Note that a Transform object does not contain any information about the extent and pixel size
    of the source or destination image, it only specifies the coordinate transformation itself.
    When a Transform object is used to transform image data (see :func:`project` and :func:`place`),
    the pixel_size information should be specified separately.
    """

    """
    Args:
        transform:
            2x2 transformation matrix that transforms (y,x)
            coordinates in the source image to (y,x) coordinates in the destination image.
            This matrix may include astropy units,
            for instance when converting from normalized SLM coordinates to physical coordinates in micrometers.
            Note:
                When specified, the units of the transform must match the units of the pixel_size of the destination
                divided by the pixel size of the source.
        source_origin:
            (y,x) coordinate of the origin in the source image, relative to the center of the image.
            By default, the center of the source image is taken as the origin of the transform,
            meaning that that point is mapped onto the destination_origin.
            Note:
                When specified, the units of source_origin must match the units of the pixel_size of the source.
        destination_origin:
            (y,x) coordinate of the origin in the destination image, relative to the center of the image.
            By default, the center of the destination image is taken as the origin of the transform,
            meaning that the source_origin is mapped to that point.
            Note:
                the units of destination_origin must match the units of the pixel_size of the source.


    """

    def __init__(
        self,
        transform: Optional[TransformType] = None,
        source_origin: Optional[CoordinateType] = None,
        destination_origin: Optional[CoordinateType] = None,
    ):
        self.transform = Quantity(transform if transform is not None else np.eye(2))
        self.source_origin = Quantity(source_origin) if source_origin is not None else None
        self.destination_origin = Quantity(destination_origin) if destination_origin is not None else None

        if source_origin is not None:
            self.destination_unit(self.source_origin.unit)  # check if the units are compatible

    def destination_unit(self, src_unit: u.Unit) -> u.Unit:
        """Computes the unit of the output of the transformation, given the unit of the input.

        Raises:
            ValueError: If src_unit does not match the unit of the source_origin (if specified) or
                if dst_unit does not match the unit of the destination_origin (if specified).
        """
        if self.source_origin is not None and not self.source_origin.unit.is_equivalent(src_unit):
            raise ValueError("src_unit must match the units of source_origin.")

        dst_unit = (self.transform[0, 0] * src_unit).unit
        if self.destination_origin is not None and not self.destination_origin.unit.is_equivalent(dst_unit):
            raise ValueError("dst_unit must match the units of destination_origin.")

        return dst_unit

    def cv2_matrix(
        self,
        source_shape: Sequence[int],
        source_pixel_size: CoordinateType,
        destination_shape: Sequence[int],
        destination_pixel_size: CoordinateType,
    ) -> np.ndarray:
        """Returns the transformation matrix in the format used by cv2.warpAffine.

        This matrix is 2x3, with the last column corresponding to the translation vector.

        Note: OpenCV uses the (x,y) convention, so the matrix is transposed compared to the numpy matrix.
        Note: OpenCV uses the _center_ of the top-left corner as the origin, whereas OpenWFS uses the centerso the origin is corrected.
            of the image as the origin. The difference of half the image size minus half the
            pixel size is corrected in the transformation matrix.
        """
        if min(source_shape) < 1 or min(destination_shape) < 1:
            raise ValueError("Image size must be positive")

        source_origin = 0.5 * (np.array(source_shape) - 1.0) * source_pixel_size
        if self.source_origin is not None:
            source_origin += self.source_origin

        destination_origin = 0.5 * (np.array(destination_shape) - 1.0) * destination_pixel_size
        if self.destination_origin is not None:
            destination_origin += self.destination_origin

        centered_transform = Transform(
            transform=self.transform,
            source_origin=source_origin,
            destination_origin=destination_origin,
        )

        # then convert the transform to a matrix, using the specified pixel sizes
        transform_matrix = centered_transform.to_matrix(
            source_pixel_size=source_pixel_size,
            destination_pixel_size=destination_pixel_size,
        )

        # finally, convert the matrix to the format used by cv2.warpAffine by swapping x and y columns and rows
        transform_matrix = transform_matrix[[1, 0, 2], 0:3]
        transform_matrix = transform_matrix[0:2, [1, 0, 2]]  # discard last row
        return transform_matrix

    def to_matrix(self, source_pixel_size: CoordinateType, destination_pixel_size: CoordinateType) -> np.ndarray:
        """Returns a homogeneous transformation matrix that transforms (y,x,1) coordinates to (y', x', 1)."""
        matrix = np.eye(3)
        matrix[0:2, 0:2] = unitless(self.transform * source_pixel_size / destination_pixel_size)
        if self.destination_origin is not None:
            matrix[0:2, 2] = unitless(self.destination_origin / destination_pixel_size)
        if self.source_origin is not None:
            matrix[0:2, 2] -= unitless((self.transform @ self.source_origin) / destination_pixel_size)
        return matrix

    def opencl_matrix(self) -> np.ndarray:
        """Returns the transformation matrix (including translation) in the format used by OpenCL.

        The returned matrix is a 3x4 matrix that includes the transformation matrix and the translation vector.
        The last two columns are for padding only.
        """

        # Compute the homogeneous transform matrix, and transpose it.
        # Then, swap x and y rows and columns,
        # and flip the sign of the y-axis
        matrix = self.to_matrix((1.0, 1.0), (1.0, 1.0)).transpose()

        # convert to opencl format: swap x and y rows and columns,
        # and flip the sign of the y-axis output.
        transform = np.eye(3, 4, dtype="float32", order="C")
        transform[:, 0] = matrix[[1, 0, 2], 1]
        transform[:, 1] = -matrix[[1, 0, 2], 0]
        return transform

    @staticmethod
    def zoom(scale: float):
        """Returns a transform that just zooms in the image."""
        return Transform(transform=np.array(((scale, 0.0), (0.0, scale))))

    def __matmul__(self, other):
        """The matrix multiplication operator is used to compose transformations,
        and to apply a transformation to a vector."""
        if isinstance(other, Transform):
            return self.compose(other)
        else:
            return self.apply(other)

    def apply(self, vector: CoordinateType) -> CoordinateType:
        """Applies the transformation to a column vector.

        If `vector` is a 2-D array, applies the transformation to each column of `vector` individually.
        """
        if self.source_origin is not None:
            vector = vector - self.source_origin
        vector = self.transform @ vector
        if self.destination_origin is not None:
            vector = vector + self.destination_origin
        return vector

    def inverse(self):
        """Compute the inverse transformation,
        such that the composition of the transformation and its inverse is the identity.
        """

        # invert the transform matrix
        if self.transform is not None:
            transform = np.linalg.inv(self.transform)
        else:
            transform = None

        # swap source and destination origins
        return Transform(
            transform,
            source_origin=self.destination_origin,
            destination_origin=self.source_origin,
        )

    def compose(self, other: "Transform"):
        """Compose two transformations.

        Args:
            other (Transform): the transformation to apply first

        Returns:
            Transform: the composition of the two transformations
        """
        transform = self.transform @ other.transform
        destination_origin = (
            self.apply(other.destination_origin) if other.destination_origin is not None else self.destination_origin
        )
        return Transform(transform, other.source_origin, destination_origin)

    def _standard_input(self) -> Quantity:
        """Construct standard input points (1,0), (0,1) and (0,0) with the source unit of this transform."""
        return Quantity(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), self.source_origin)

    @classmethod
    def identity(cls):
        return Transform()


def place(
    out_shape: tuple[int, ...],
    out_pixel_size: Quantity,
    source: np.ndarray,
    offset: Optional[Quantity] = None,
    out: Optional[np.ndarray] = None,
):
    """Takes a source array and places it in an otherwise empty array of specified shape and pixel size.

    The source array must have a pixel_size property (see set_pixel_size).
    It is scaled (using area interpolation) so that the pixel size matches that of the output.
    Then, the source array is placed into the output array at the position given by offset,
    where `offset=None` or `offset=(0.0, 0.0) * pixel_size` corresponds to centering the source image
    with respect to the output array.
    Parts of the source array that extend beyond the output are cropped, and parts of the array that are not
    covered by the source array are zero padded.

    Note: this function currently works for 2-D inputs only

    Args:

    """
    out_extent = out_pixel_size * np.array(out_shape)
    transform = Transform(destination_origin=offset)
    return project(source, out_extent=out_extent, out_shape=out_shape, transform=transform, out=out)


def project(
    source: np.ndarray,
    *,
    source_extent: Optional[ExtentType] = None,
    transform: Optional[Transform] = None,
    out: Optional[np.ndarray] = None,
    out_extent: Optional[ExtentType] = None,
    out_shape: Optional[tuple[int, ...]] = None
) -> np.ndarray:
    """Projects the input image onto an array with specified shape and resolution.

    The input image is scaled so that the pixel sizes match those of the output,
    and cropped/zero-padded so that the data shape matches that of the output.
    Optionally, an additional :class:`~Transform` can be specified, e.g., to scale or translate the source image.

    Args:
        source: input image.
        source_extent: extent of the source image in some physical unit.
            If not given (``None``), the extent metadata of the input image is used.
            see :func:`~get_extent`.
        transform: optional transformed (rotate, translate, etc.)
            to appy to the source image before placing it in the output
        out: optional array where the output image is stored in.
        out_extent: extent of the output image in some physical unit.
            If not given, the extent metadata of the out image is used.
        out_shape: shape of the output image.
            This value is ignored if `out` is specified.

    Returns:
        np.ndarray: the projected image (`out` if specified, otherwise a new array)
    """
    transform = transform if transform is not None else Transform()
    if out is not None:
        if out_shape is not None and out_shape != out.shape:
            raise ValueError("out_shape and out.shape must match. Note that out_shape may be omitted")
        if out.dtype != source.dtype:
            raise ValueError("out and source must have the same dtype")
        out_shape = out.shape
        out_extent = out_extent or get_extent(out)
    if out_shape is None:
        raise ValueError("Either out_shape or out must be specified")
    if out_extent is None:
        raise ValueError("Either out_extent or the pixel_size metadata of out must be specified")
    source_extent = source_extent if source_extent is not None else get_extent(source)
    source_ps = source_extent / np.array(source.shape)
    out_ps = out_extent / np.array(out_shape)

    t = transform.cv2_matrix(source.shape, source_ps, out_shape, out_ps)
    # swap x and y in matrix and size, since cv2 uses the (x,y) convention.
    out_size = (out_shape[1], out_shape[0])
    if (source.dtype == np.complex128) or (source.dtype == np.complex64):
        if out is None:
            out = np.zeros(out_shape, dtype=source.dtype)
        # real part
        out.real = cv2.warpAffine(
            source.real,
            t,
            out_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
        # imaginary part
        out.imag = cv2.warpAffine(
            source.imag,
            t,
            out_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

    else:
        dst = cv2.warpAffine(
            source,
            t,
            out_size,
            dst=out,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
        if out is not None and out is not dst:
            raise ValueError("OpenCV did not use the specified output array. This should not happen.")
        out = dst
    return set_pixel_size(out, out_ps)


def set_pixel_size(data: ArrayLike, pixel_size: Optional[Quantity]) -> np.ndarray:
    """
    Sets the pixel size metadata for the given data array.

    Args:
        data (ArrayLike): The input data array.
        pixel_size (Optional[Quantity]): The pixel size to be set. When a single-element pixel size is given,
            it is broadcasted to all dimensions of the data array.
            Passing None sets the pixel size metadata to None.

    Returns:
        np.ndarray: The modified data array with the pixel size metadata.

    Usage:
    >>> data = np.array([[1, 2], [3, 4]])
    >>> pixel_size = 0.1 * u.m
    >>> modified_data = set_pixel_size(data, pixel_size)
    """
    data = np.array(data)

    if pixel_size is not None and pixel_size.size == 1:
        pixel_size = pixel_size * np.ones(data.ndim)

    data.dtype = np.dtype(data.dtype, metadata={"pixel_size": pixel_size})
    return data


def get_pixel_size(data: np.ndarray) -> Optional[Quantity]:
    """
    Extracts the pixel size metadata from the data array.

    Args:
        data (np.ndarray): The input data array or Quantity.

    Returns:
        OptionalQuantity]: The pixel size metadata, or None if no pixel size metadata is present.

    Usage:
    >>> import astropy.units as u
    >>> import numpy as np
    >>> data = set_pixel_size(((1, 2), (3, 4)), 5 * u.um)
    >>> pixel_size = get_pixel_size(data)

    TODO: return 1 if no pixel size is present
    """
    metadata = data.dtype.metadata
    if metadata is None:
        return None
    return data.dtype.metadata.get("pixel_size", None)


def get_extent(data: np.ndarray) -> Quantity:
    """
    Extracts the extent from the data array.
    The extent is always equal to `shape * pixel_size`.
    """
    pixel_size = get_pixel_size(data)
    if pixel_size is None:
        return Quantity(data.shape)
    return data.shape * pixel_size


def set_extent(data: np.ndarray, extent: ExtentType) -> np.ndarray:
    """
    Sets the extent metadata for the given data array.

    Args:
        data (ArrayLike): The input data array.
        extent (ArrayLike): The extent to be set. When a single-element extent is given,
            it is broadcasted to all dimensions of the data array.
            Passing None sets the extent metadata to None.

    Returns:
        np.ndarray: The modified data array with the extent metadata.

    Usage:
    >>> data = np.array([[1, 2], [3, 4]])
    >>> extent = 0.1 * u.m
    >>> modified_data = set_extent(data, extent)
    """
    return set_pixel_size(data, extent / np.array(data.shape))

def Transformation_Matrix_SLM_and_stage_to_World_Coordinates(Data, stage_disp_xy_um=1, SLM_grad_xy=1):
    """
    In Experimental setups we are often dealing with several different coordinate system. In a setup with an SLM 
    projecting a pattern onto a lens, we need to be able to convert coordinates between the SLM coordinate system and the 
    world coordinate system of the experimental setup.

    This function provides two callbiration matrices, G and M, as defined in https://doi.org/10.1364/OL.400985, to 
    convert coordinates between the coodinate system of the SLM and the global coordinate system of the experimental setup.

    Input: 
    Data: numpy array of size N by M by 8. 
        im1: reference image
        im2: image after moving the stage in +x direction by stage_disp_xy_um
        im3: reference image
        im4: image after moving the stage in +y direction by stage_disp_xy_um
        im5: reference image
        im6: image after applying a slope in +x direction to the SLM
        im7: reference image
        im8: image after applying a slope in +y direction to the SLM
    stage_disp_xy_um: scalar giving the displacement of the stage in um used for im2 and im4
    SLM_grad_xy: scalar giving the magnitude of the slope applied to the SLM in rad/SLM pizels used for im6 and im8

    Output:
        G: conversion matrix [TPM frame pixels*SLM pixels]
        M: conversion matrix [TPM frame pixels/um]
    """

    # Type checks
    if not isinstance(Data, np.ndarray):
        raise TypeError(f"Data must be a np.ndarray, got {type(Data)}")
    if not isinstance(stage_disp_xy_um, (int, float)):
        raise TypeError(f"stage_distance_um must be a number, got {type(stage_disp_xy_um)}")
    if not isinstance(SLM_grad_xy, (int, float)):
        raise TypeError(f"delta_slope must be a number, got {type(SLM_grad_xy)}")
    
    # Build matrix (M) that relates coordinates of the Zaber stage to coordinates in the TPM frame
    (dy1, dx1) = find_shift(Data[:,:,0],Data[:,:,1])
    (dy2, dx2) = find_shift(Data[:,:,2],Data[:,:,3])

    # Make a 2×2 matrix where each column is a shift vector
    M0 = np.array([[dx1, dx2],
                [dy1, dy2]])

    stage_displacement = [stage_disp_xy_um, stage_disp_xy_um]
  
    M = M0 / stage_displacement   # solving the system M0 = M * stage_displacement

    # Build matrix (G) that relates coordinates of the SLM to coordinates in the TPM frame
    (dy1, dx1) = find_shift(Data[:,:,4],Data[:,:,5])
    (dy2, dx2) = find_shift(Data[:,:,6],Data[:,:,7])

    # Make a 2×2 matrix where each column is a shift vector
    G0 = np.array([[dx1, dx2],
                [dy1, dy2]])

    SLM_grad = [SLM_grad_xy, SLM_grad_xy]

    G = G0 / SLM_grad   # element-wise division, solving the system G0 = G * SLM_grad

    return M, G


def cross_correlation_mean_corrected(f,g):
    """
    This function computes the cross-correlation between two arrays using the Cross-Correlation Theorem.
    See: https://mathworld.wolfram.com/Cross-CorrelationTheorem.html
    DC component is removed from both arrays before computation.

    Input:
    f (np.ndarray): First input ND array.
    g (np.ndarray): Second input ND array.

    Output:
    np.ndarray: Cross-correlation of f and g.
    """

    # Check if inputs are numpy arrays with the same size
    if not (isinstance(f, np.ndarray) and isinstance(g, np.ndarray)):
        raise TypeError("Both inputs must be numpy arrays.")
    if f.shape != g.shape:
        raise ValueError("Both inputs must have the same shape.")
    if not (np.isrealobj(f) and np.isrealobj(g)):
        raise ValueError("Both inputs must be real-valued arrays.")
    
    # remove average of f and g
    df = f - np.mean(f)
    dg = g - np.mean(g)

    # check if arrays are all zeros after mean subtraction, because then cross correlati
    if np.all(df == 0) or np.all(dg == 0):
        raise ValueError("After mean correction, one of the arrays is all zeros (constant input).")

    # Cross-correlation via FFT. Take real part to avoid numerical noise.
    cross_correlation = np.real(np.fft.ifftn(np.conj(np.fft.fftn(df)) * np.fft.fftn(dg))) 

    return cross_correlation


def find_shift(im1, im2):
    """
    Find the shift between two ND arrays using cross-correlation. 
    The function computes the cross-correlation between the two arrays and identifies the peak to determine the shift.

    Input:
    im1 (np.ndarray): First input ND array.
    im2 (np.ndarray): Second input ND array.   

    Output:
    Shift as a tuple of integers. first the shift along the first dimension (rows), then the second dimension (columns) etc...
    Important, first direction is vertical, second horizontal!!!, so corresoponding to (y,x,...) using the default coordinate system. 

    Important:
    Periodicity of the array is assumed, so the returned shift has a maximum of half the array size along each dimension. 
    This may cause incorrect results if the actual shift is larger than half the array size.
    """    
    cc = cross_correlation_mean_corrected(im1, im2)

    # find index of maximum correlation
    shift = np.array(np.unravel_index(np.argmax(cc), cc.shape))

    # convert to signed shifts for periodic boundary
    for i in range(len(shift)):
        if shift[i] > cc.shape[i] // 2:
            shift[i] -= cc.shape[i]

    return tuple(shift)