import matplotlib.pyplot as plt
import astropy.units as u
from .core import Detector
import asyncio


def grab_and_show(cam: Detector, magnification=1.0):
    image = cam.read()
    dimensions = cam.extent().to_value(u.um) / magnification
    plt.imshow(image, extent=(0.0, dimensions[0], 0.0, dimensions[1]), cmap='gray')
    plt.xlabel('μm')
    plt.ylabel('μm')
