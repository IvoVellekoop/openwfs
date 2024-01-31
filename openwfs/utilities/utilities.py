from astropy import units as u
from astropy.units import Quantity
from numpy.typing import ArrayLike
from typing import Union, Sequence, Optional
import numpy as np
import cv2
from dataclasses import dataclass

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
        For example, ```np.array(1 * u.s / u.ms) == 1```.
        Whereas `unitless(1 * u.s / u.ms) == 1000` gives the correct answer.

    Usage:
    >>> data = np.array([1.0, 2.0, 3.0]) * u.m
    >>> unitless_data = unitless(data)
    """
    if isinstance(data, Quantity):
        return data.to_value(u.dimensionless_unscaled)
    else:
        return np.array(data)


@dataclass
class Transform:
    """Represents a transformation from one coordinate system to the other.

    Transform objects are used to specify any combination of shift, (anisotropic) scaling, rotation and shear.
    Elements of the transformation are specified with an astropy.unit attached.
    """

    """
    Args:
        transform:
            2x2 transformation matrix that transforms (y,x)
            coordinates in the source image to (y,x) coordinates in the destination image.
            This matrix may include astropy units,
            for instance when converting from normalized SLM coordinates to physical coordinates in micrometers.
            Note:
                the units of the transform must match the units of the pixel_size of the destination
                divided by the pixel size of the source.
        source_origin:
            (y,x) coordinate of the origin in the source image, relative to the center of the image.
            By default, the center of the source image is taken as the origin of the transform,
            meaning that that point is mapped onto the destination_origin.
            Note:
                the units of source_origin must match the units of the pixel_size of the source.
        destination_origin:
            (y,x) coordinate of the origin in the destination image, relative to the center of the image.
            By default, the center of the destination image is taken as the origin of the transform,
            meaning that the source_origin is mapped to that point.
            Note:
                the units of destination_origin must match the units of the pixel_size of the source.


    """

    def __init__(self, transform: Optional[TransformType] = None,
                 source_origin: Optional[CoordinateType] = None,
                 destination_origin: Optional[CoordinateType] = None):

        self.transform = Quantity(transform if transform is not None else np.eye(2))
        self.source_origin = Quantity(source_origin) if source_origin is not None else None
        self.destination_origin = Quantity(destination_origin) if destination_origin is not None else None

    def cv2_matrix(self, source_shape: Sequence[int],
                   source_pixel_size: CoordinateType,
                   destination_shape: Sequence[int],
                   destination_pixel_size: CoordinateType) -> np.ndarray:
        """Returns the transformation matrix in the format used by cv2.warpAffine."""

        # first construct a transform that is relative to the center of the image, as required by cv2.warpAffine
        source_origin = 0.5 * np.array(source_shape) * source_pixel_size
        if self.source_origin is not None:
            source_origin += self.source_origin

        destination_origin = 0.5 * np.array(destination_shape) * destination_pixel_size
        if self.destination_origin is not None:
            destination_origin += self.destination_origin

        centered_transform = Transform(transform=self.transform,
                                       source_origin=source_origin,
                                       destination_origin=destination_origin)

        # then convert the transform to a matrix, using the specified pixel sizes
        transform_matrix = centered_transform.to_matrix(source_pixel_size=source_pixel_size,
                                                        destination_pixel_size=destination_pixel_size)

        # finally, convert the matrix to the format used by cv2.warpAffine by swapping x and y columns and rows
        transform_matrix = transform_matrix[[1, 0], :]
        transform_matrix = transform_matrix[:, [1, 0, 2]]
        return transform_matrix

    def to_matrix(self, source_pixel_size: CoordinateType, destination_pixel_size: CoordinateType) -> np.ndarray:
        matrix = np.zeros((2, 3))
        matrix[0:2, 0:2] = unitless(self.transform * source_pixel_size / destination_pixel_size)
        if self.destination_origin is not None:
            matrix[0:2, 2] = unitless(self.destination_origin / destination_pixel_size)
        if self.source_origin is not None:
            matrix[0:2, 2] -= unitless((self.transform @ self.source_origin) / destination_pixel_size)
        return matrix

    def opencl_matrix(self) -> np.ndarray:
        # compute the transform for points (0,1), (1,0), and (0, 0)
        matrix = self.to_matrix((1.0, 1.0), (1.0, 1.0))

        # subtract the offset (transform of 0,0) from the first two points
        # to construct the homogeneous transformation matrix
        # convert to opencl format: swap x and y columns (note: the rows were
        # already swapped in the construction of t2), and flip the sign of the y-axis.
        transform = np.eye(3, 4, dtype='float32', order='C')
        transform[0, 0:3] = matrix[1, [1, 0, 2],]
        transform[1, 0:3] = -matrix[0, [1, 0, 2],]
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

         If `vector` is a 2-D array, applies the transformation to each column of `vector` individually."""
        if self.source_origin is not None:
            vector = vector - self.source_origin
        vector = self.transform @ vector
        if self.destination_origin is not None:
            vector = vector + self.destination_origin
        return vector

    def inverse(self):
        """Compute the inverse transformation,
        such that the composition of the transformation and its inverse is the identity."""

        # invert the transform matrix
        if self.transform is not None:
            transform = np.linalg.inv(self.transform)
        else:
            transform = None

        # swap source and destination origins
        return Transform(transform, source_origin=self.destination_origin, destination_origin=self.source_origin)

    def compose(self, other: 'Transform'):
        """Compose two transformations.

        Args:
            other (Transform): the transformation to apply first

        Returns:
            Transform: the composition of the two transformations
        """
        transform = self.transform @ other.transform
        source_origin = other.source_origin
        destination_origin = self.apply(other.destination_origin) if other.destination_origin is not None else None
        return Transform(transform, source_origin, destination_origin)

    def _standard_input(self) -> Quantity:
        """Construct standard input points (1,0), (0,1) and (0,0) with the source unit of this transform.."""
        return Quantity(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), self.source_origin)

    @classmethod
    def identity(cls):
        return Transform()


def place(out_shape: Sequence[int], out_pixel_size: Quantity, source: np.ndarray, offset: Optional[Quantity] = None,
          out: Optional[np.ndarray] = None):
    """Takes a source array and places it in an otherwise empty array of specified shape and pixel size.

    The source array must have a pixel_size property (see set_pixel_size).
    It is scaled (using area interpolation) so that the pixel size matches that of the output.
    Then, the source array is placed into the output array at the position given by offset,
    where `offset=None` or `offset=(0.0, 0.0) * pixel_size` corresponds to centering the source image
    with respect to the output array.
    Parts of the source array that extend beyond the output are cropped, and parts of the array that are not
    covered by the source array are zero padded.

    Note: this function currently works for 2-d inputs only

    Args:

    """
    transform = Transform(destination_origin=offset)
    return project(out_shape, out_pixel_size, source, transform, out)


def project(out_shape: Sequence[int], out_pixel_size: Quantity, source: np.ndarray,
            transform: Transform, out: Optional[np.ndarray] = None):
    """Projects the input image onto an array with specified shape and resolution.

    The input image is scaled so that the pixel sizes match those of the output,
    and cropped/zero-padded so that the data shape matches that of the output.
    Optionally, an additional transformation can be specified, e.g., to scale or translate the source image.
    This transformation is specified as a 2x3 transformation matrix in homogeneous coordinates.

    Args:
        out_shape (tuple[int, int]): number of pixels in the output array
        out_pixel_size (Quantity): pixel size of the output array, 2-element array
        source (np.ndarray): input image.
            Must have the pixel_size set (see set_pixel_size)
        transform: transformation to appy to the source image before placing it in the output
        out (np.ndarray): optional array where the output image is stored in.
            If specified, `out_shape` is ignored.

    Returns:

    """
    if transform is None:
        transform = Transform()
    t = transform.cv2_matrix(source.shape, get_pixel_size(source), out_shape, out_pixel_size)
    # swap x and y in matrix and size, since cv2 uses the (x,y) convention.
    out_size = (out_shape[1], out_shape[0])
    dst = cv2.warpAffine(source, t, out_size, dst=out, flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0,))
    return set_pixel_size(dst, out_pixel_size)


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

    data.dtype = np.dtype(data.dtype, metadata={'pixel_size': pixel_size})
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
    """
    metadata = data.dtype.metadata
    if metadata is None:
        return None
    return data.dtype.metadata.get('pixel_size', None)
