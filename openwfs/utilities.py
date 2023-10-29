import matplotlib.pyplot as plt
import astropy.units as u
from .core import DataSource


def grab_and_show(cam: DataSource, magnification=1.0):
    cam.trigger()
    image = cam.read()
    dimensions = cam.dimensions().to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')
