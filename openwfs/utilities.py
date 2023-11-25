import matplotlib.pyplot as plt
import astropy.units as u
from astropy.units import Quantity
from .core import Detector, get_pixel_size
from typing import Union, Sequence
import numpy as np
import cv2


def grab_and_show(cam: Detector, magnification=1.0):
    image = cam.read()
    dimensions = cam.extent.to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')


def project(out_shape: Sequence[int], out_pixel_size: Quantity, source: np.ndarray,
            transform: Union[np.ndarray, float, None] = None, out: Union[np.ndarray, None] = None):
    """Projects the input image onto an array with specified shape and resolution.

    The input image is scaled so that the pixel sizes match those of the output,
    and cropped/zero-padded so that the data shape matches that of the output.
    Optionally, an additional transformation can be specified, e.g. to scale or translate the source image.
    This transformation is specified as a scalar (scaling only), or a 2x3 transformation matrix in homogeneous coordinates.

    Args:
        out_shape (tuple[int,int]): number of pixels in the output array
        out_pixel_size (Quantity): pixel size of the output array, 2-element array
        source (np.ndarray): input image.
            Must have the pixel_size set (see set_pixel_size)
        transform (nd.ndarray|float|None): optional transformation to appy to the input image
        out (np.ndarray): optional array where the output image is stored in.
            If specified, `out_shape` is ignored.

    Returns:

    """
    if transform is None:
        transform = np.eye(2, 3)
    elif np.size(transform) == 1:
        transform = np.eye(2, 3) * transform

    transform = transform * float(out_pixel_size / get_pixel_size(source))
    return cv2.warpAffine(source, transform, out_shape, dst=out, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0,))
