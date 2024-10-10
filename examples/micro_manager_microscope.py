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
from openwfs.processors import SingleRoi
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
image_data[256, 256] = 100
src = StaticSource(
    data=image_data,
    pixel_size=specimen_pixel_size,
)
aberrations = StaticSource(extent=2 * numerical_aperture)

slm = SLM(shape=(100, 100), field_amplitude=gaussian((100, 100), waist=1.0))

# Create a microscope with the given parameters
mic = Microscope(
    source=src,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    wavelength=wavelength,
    aberrations=aberrations,
    incident_field=slm.field,
)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = Camera(
    mic,
    analog_max=None,
    shot_noise=True,
    digital_max=255,
    data_shape=camera_resolution,
    pixel_size=camera_pixel_size,
)

feedback = SingleRoi(source=cam, pos=(256, 256), mask_type="gaussian", radius=3.0, waist=1.0)

alg = WFSController(
    FourierDualReference, feedback=feedback, slm=sim.slm, slm_shape=slm_shape, k_radius=3.5, phase_steps=5
)

# construct dictionary of objects to expose to Micro-Manager
devices = {"camera": cam, "stage": mic.xy_stage, "algorithm": alg}
