from astropy import units as u
from matplotlib import pyplot as plt

from .core import Detector
from .utilities import get_pixel_size


def grab_and_show(cam: Detector, magnification=1.0):
    image = cam.read()
    dimensions = cam.extent.to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')


def imshow(data):
    pixel_size = get_pixel_size(data)
    if pixel_size is not None:
        extent = pixel_size * data.shape
        plt.xlabel(extent.unit)
        plt.ylabel(extent.unit)
        extent = extent.value
    else:
        extent = data.shape
    plt.imshow(data, extent=(0.0, extent[0], 0.0, extent[1]), cmap='gray')
    plt.colorbar()
    plt.show()
    plt.pause(0.1)
