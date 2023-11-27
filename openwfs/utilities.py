import matplotlib.pyplot as plt
import astropy.units as u
from astropy.units import Quantity
from .core import Detector, get_pixel_size, set_pixel_size
from typing import Union, Sequence
import numpy as np
import cv2


def grab_and_show(cam: Detector, magnification=1.0):
    image = cam.read()
    dimensions = cam.extent.to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')


def place(out_shape: Sequence[int], out_pixel_size: Quantity, source: np.ndarray, offset: Union[Quantity, None] = None,
          out: Union[np.ndarray, None] = None):
    """Takes a source array and places it in an otherwise empty array of specified shape and pixel size.

    The source array must have a pixel_size property (see set_pixel_size).
    It is scaled (using area interpolation) so that the pixel size matches that of the output.
    Then, the source array is placed into the output array at the position given by offset,
    where `offset=None` or `offset=(0.0, 0.0) * pixel_size` corresponds to centering the source image
    with respect to the output array.
    Parts of the source array that extend beyond the output are cropped, and parts of the array that are not
    covered by the source array are zero padded.

    Args:

    """
    transform = np.eye(2, 3)
    src_pixel_size = get_pixel_size(source)
    if offset is not None:
        transform[0:2, 2] = offset / src_pixel_size

    return project(out_shape, out_pixel_size, source, transform, out)


def project(out_shape: Sequence[int], out_pixel_size: Quantity, source: np.ndarray,
            transform: np.ndarray, out: Union[np.ndarray, None] = None):
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
    # matrix to adjust pixel size
    scale = np.diag((out_pixel_size / get_pixel_size(source)).to_value(u.dimensionless_unscaled))

    # adjust center positions (cv2 takes the corner as origin, not the center)
    transform[0:2, 2] += 0.5 * (np.array(out_shape) - scale @ np.array(source.shape))

    # swap x and y in matrix and size, since cv2 uses the (x,y) convention.
    transform = np.array(((transform[1, 1], transform[1, 0], transform[1, 2]),
                          (transform[0, 1], transform[0, 0], transform[0, 2])))
    out_size = (out_shape[1], out_shape[0])
    dst = cv2.warpAffine(source, scale @ transform, out_size, dst=out, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0,))
    return set_pixel_size(dst, out_pixel_size)
