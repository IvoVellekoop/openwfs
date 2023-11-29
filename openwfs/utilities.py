import matplotlib.pyplot as plt
import astropy.units as u
from astropy.units import Quantity
from .core import Detector, get_pixel_size, set_pixel_size
from typing import Union, Sequence, Optional
import numpy as np
import cv2


def grab_and_show(cam: Detector, magnification=1.0):
    image = cam.read()
    dimensions = cam.extent.to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')


def imshow(data):
    pixel_size = get_pixel_size(data, may_fail=True)
    extent = pixel_size * data.shape
    plt.xlabel(extent.unit)
    plt.ylabel(extent.unit)
    extent = extent.value
    plt.imshow(data, extent=(0.0, extent[0], 0.0, extent[1]), cmap='gray')
    plt.show()
    plt.pause(0.1)


class Transform:
    """Represents a transformation from one coordinate system to the other.

    Transform objects are used to specify any combination of shift, (anisotropic) scaling, rotation and shear.
    Elements of the transformation are specified with an astropy.unit attached.
    """

    def __init__(self, transform: Union[np.ndarray, Quantity, Sequence[Sequence[float]]] = ((1.0, 0.0), (0.0, 1.0)),
                 source_origin: Union[Sequence[float], np.ndarray, Quantity, None] = None,
                 destination_origin: Union[Sequence[float], np.ndarray, Quantity, None] = None):
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
        self.transform = Quantity(transform)
        self.source_origin = None if source_origin is None else Quantity(source_origin)
        self.destination_origin = None if destination_origin is None else Quantity(destination_origin)

    def cv2_matrix(self, source_shape, source_pixel_size, destination_shape, destination_pixel_size):
        transform = np.zeros((2, 3))

        # compute the transform itself, from source pixels to destination pixels
        # this also verifies that the transform has the correct pixel sizes
        # note: x and y are swapped to have a matrix in cv2's (x,y) convention
        transform[0, 0] = self.transform[1, 1] * source_pixel_size[1] / destination_pixel_size[1]
        transform[0, 1] = self.transform[1, 0] * source_pixel_size[0] / destination_pixel_size[1]
        transform[1, 0] = self.transform[0, 1] * source_pixel_size[1] / destination_pixel_size[0]
        transform[1, 1] = self.transform[0, 0] * source_pixel_size[0] / destination_pixel_size[0]

        # apply the transform to the source offset
        # compute source offset in pixels
        # this also verifies that source_origin and source_pixel_size have compatible units
        transform[0, 2] = - 0.5 * source_shape[1]
        transform[1, 2] = - 0.5 * source_shape[0]
        if self.source_origin is not None:
            transform[0, 2] -= self.source_origin[1] / source_pixel_size[1]
            transform[1, 2] -= self.source_origin[0] / source_pixel_size[0]
        transform[0:2, 2] = transform[0:2, 0:2] @ transform[0:2, 2]

        # apply the destination offset
        transform[0, 2] += 0.5 * destination_shape[1]
        transform[1, 2] += 0.5 * destination_shape[0]
        if self.destination_origin is not None:
            transform[0, 2] += self.destination_origin[1] / destination_pixel_size[1]
            transform[1, 2] += self.destination_origin[0] / destination_pixel_size[0]

        return transform

    @staticmethod
    def zoom(scale: float):
        """Returns a transform that just zooms in the image."""
        return Transform(transform=((scale, 0.0), (0.0, scale)))

    @staticmethod
    def identity():
        """Returns a transform that does nothing."""
        return Transform.zoom(1.0)


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
            transform: Transform, out: Union[np.ndarray, None] = None):
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
        transform = Transform.identity()
    t = transform.cv2_matrix(source.shape, get_pixel_size(source), out_shape, out_pixel_size)
    # swap x and y in matrix and size, since cv2 uses the (x,y) convention.
    out_size = (out_shape[1], out_shape[0])
    dst = cv2.warpAffine(source, t, out_size, dst=out, flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0,))
    return set_pixel_size(dst, out_pixel_size)
