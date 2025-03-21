"""Micro-Manager simulated microscope
======================================================================
This script simulates a microscope with a random noise image as a mock specimen.
The numerical aperture, stage position, and other parameters can be modified through the Micro-Manager GUI.

To use it:
  * make sure  you have the PyDevice adapter installed in Micro-Manager (install the nightly build if you don't have it).
  * load the micro_manager_microscope.cfg hardware configuration in Micro-Manager,
  * locate the micro_manager_microscope.py in the file open dialog box that popps up.
  * take a snapshot or turn on live preview, you may need to auto-adjust the color scale
  * experiment with the numerical aperture or set the stage X and Y position in the device properties.

See the 'Sample Microscope' example for a microscope simulation that runs from Python directly.
"""

import astropy.units as u
import numpy as np

from openwfs.simulation import Microscope, StaticSource, Camera, SLM
from openwfs.utilities.patterns import gaussian

specimen_resolution = (512, 512)  # height × width in pixels of the specimen image
specimen_pixel_size = 120 * u.nm  # resolution (pixel size) of the specimen image
magnification = 40  # magnification from object plane to camera.
numerical_aperture = 0.85  # numerical aperture of the microscope objective
wavelength = 532.8 * u.nm  # wavelength of the light, for computing diffraction.
camera_resolution = (256, 256)  # number of pixels on the camera
camera_pixel_size = 6.45 * u.um  # Size of the pixels on the camera

# Create a random noise image with a few bright spots
image_data = np.maximum(np.random.randint(-10000, 100, specimen_resolution, dtype=np.int16), 0)
src = StaticSource(
    data=image_data,
    pixel_size=specimen_pixel_size,
)

slm = SLM(shape=(100, 100), field_amplitude=gaussian((100, 100), waist=1.0))

# Create a microscope with the given parameters
mic = Microscope(
    source=src,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    wavelength=wavelength,
    incident_field=slm.field,
)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = Camera(mic, analog_max=None, shot_noise=True, digital_max=255)

# construct dictionary of objects to expose to Micro-Manager
devices = {"camera": cam, "stage": mic.xy_stage}
