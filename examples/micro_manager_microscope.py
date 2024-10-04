""" Micro-Manager simulated microscope
======================================================================
This script simulates a microscope with a random noise image as a mock specimen.
The numerical aperture, stage position, and other parameters can be modified through the Micro-Manager GUI.
To use this script as a device in Micro-Manager, make sure you have the PyDevice adapter installed and
select this script in the hardware configuration wizard for the PyDevice component.

See the 'Sample Microscope' example for a microscope simulation that runs from Python directly.
"""

import astropy.units as u
import numpy as np

from openwfs.simulation import Microscope, StaticSource, Camera

specimen_resolution = (1024, 1024)  # height × width in pixels of the specimen image
specimen_pixel_size = 60 * u.nm  # resolution (pixel size) of the specimen image
magnification = 40  # magnification from object plane to camera.
numerical_aperture = 0.85  # numerical aperture of the microscope objective
wavelength = 532.8 * u.nm  # wavelength of the light, for computing diffraction.
camera_resolution = (256, 256)  # number of pixels on the camera
camera_pixel_size = 6.45 * u.um  # Size of the pixels on the camera

# Create a random noise image with a few bright spots
src = StaticSource(
    data=np.maximum(np.random.randint(-10000, 100, specimen_resolution, dtype=np.int16), 0),
    pixel_size=specimen_pixel_size,
)

# Create a microscope with the given parameters
mic = Microscope(
    source=src,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    wavelength=wavelength,
)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = Camera(
    mic,
    shot_noise=True,
    digital_max=255,
    data_shape=camera_resolution,
    pixel_size=camera_pixel_size,
)

# construct dictionary of objects to expose to Micro-Manager
devices = {"camera": cam, "stage": mic.xy_stage}
