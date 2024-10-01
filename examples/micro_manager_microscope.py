""" Sample microscope
=======================
This script simulates a microscopic imaging system, generating a random noise image as a mock source and capturing it
through a microscope with adjustable magnification, numerical aperture, and wavelength. It visualizes the original and
processed images dynamically, demonstrating how changes in optical parameters affect image quality and resolution.

This script should be opened from the μManager microscope GUI software using the PyDevice plugin.
To do so, add a PyDevice adapter to the μManager hardware configuration, and select this script as the device script.
"""

import astropy.units as u
import numpy as np

from openwfs.simulation import Microscope, StaticSource

# height × width, and resolution, of the specimen image
specimen_size = (1024, 1024)
specimen_resolution = 60 * u.nm

# magnification from object plane to camera.
magnification = 40

# numerical aperture of the microscope objective
numerical_aperture = 0.85

# wavelength of the light, for computing diffraction.
wavelength = 532.8 * u.nm

# Size of the pixels on the camera
pixel_size = 6.45 * u.um

# number of pixels on the camera
camera_resolution = (256, 256)

# Create a random noise image with a few bright spots
src = StaticSource(
    data=np.maximum(np.random.randint(-10000, 100, specimen_size, dtype=np.int16), 0),
    pixel_size=specimen_resolution,
)

# Create a microscope with the given parameters
mic = Microscope(
    source=src,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    wavelength=wavelength,
)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = mic.get_camera(
    shot_noise=True,
    digital_max=255,
    data_shape=camera_resolution,
    pixel_size=pixel_size,
)

# expose the xy-stage of the microscope
stage = mic.xy_stage

# construct dictionary of objects to expose to Micro-Manager
devices = {"camera": cam, "stage": stage}
