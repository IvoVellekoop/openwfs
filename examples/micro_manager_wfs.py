""" Micro-Manager simulated wavefront shaping
======================================================================
This script simulates a full wavefront shaping experiment in the Micro-Manager GUI.
To use it:
  * make sure  you have the PyDevice adapter installed in Micro-Manager (install the nightly build if you don't have it).
  * load the micro_manager_wfs.cfg hardware configuration in Micro-Manager,
  * locate the micro_manager_wfs.py in the file open dialog box that popps up.
  * take an initial snapshot, you may need to auto-adjust the color scale
  * to run the wavefront shaping algorithm, open the device properties, and use the 'wavefront' dropdown to select 'Optimized'
  * the Micro-Manager GUI will freeze while the algorithm is executed
  * afterwards, you can take another snapshot to see the optimized image with an improved contrast and resolution.
  * you can turn on live mode and switch between Flat and Optimized wavefronts.

You can also run this script from Python directly, in which case it will display the images in a matplotlib window.
"""

import astropy.units as u
import numpy as np
import skimage.data

from openwfs.algorithms import FourierDualReference
from openwfs.algorithms.utilities import WFSController
from openwfs.processors import SingleRoi
from openwfs.simulation import Microscope, StaticSource, Camera, SLM
from openwfs.utilities.patterns import gaussian

specimen_resolution = (512, 512)  # height × width in pixels of the specimen image
specimen_pixel_size = 60 * u.nm  # resolution (pixel size) of the specimen image
magnification = 40  # magnification from object plane to camera.
numerical_aperture = 0.85  # numerical aperture of the microscope objective
wavelength = 532.8 * u.nm  # wavelength of the light, for computing diffraction.
camera_resolution = (256, 256)  # number of pixels on the camera
camera_pixel_size = 6.45 * u.um  # Size of the pixels on the camera
slm_shape = (100, 100)

# Create a random noise image with a few bright spots, one in the center
image_data = np.maximum(np.random.randint(-10000, 100, specimen_resolution, dtype=np.int16), 0)
image_data[256, 256] = 100
src = StaticSource(
    data=image_data,
    pixel_size=specimen_pixel_size,
)

# create an aberration pattern
aberrations = StaticSource(data=skimage.data.camera() * (np.pi / 128), extent=2 * numerical_aperture)

# simulate a spatial light modulator illuminated by a gaussian beam
slm = SLM(slm_shape, field_amplitude=gaussian((100, 100), waist=1.0))

# Create a microscope
mic = Microscope(
    source=src,
    data_shape=specimen_resolution,
    magnification=magnification,
    numerical_aperture=numerical_aperture,
    wavelength=wavelength,
    aberrations=aberrations,
    incident_field=slm.field,
)

# simulate shot noise in a 16-bit camera:
saturation = mic.read().mean() * 20
cam = Camera(
    mic,
    analog_max=saturation,
    shot_noise=True,
    digital_max=0xFFFF,
)

# integrate the signal in a gaussian-weighted region of interest
feedback = SingleRoi(source=cam, pos=(256, 256), mask_type="gaussian", radius=3.0, waist=1.0)

# the WFS algorithm
alg = WFSController(FourierDualReference, feedback=feedback, slm=slm, slm_shape=slm_shape, k_radius=3.5, phase_steps=5)

# construct dictionary of objects to expose to Micro-Manager
devices = {"camera": cam, "stage": mic.xy_stage, "algorithm": alg}

if __name__ == "__main__":
    # run the algorithm
    # NOTE: the _psf field will later be replaced by a .psf field that returns a camera
    import matplotlib.pyplot as plt

    before = mic.read()
    before_psf = mic._psf
    plt.subplot(2, 3, 1)
    plt.imshow(before)
    plt.colorbar()
    plt.title("Image - no WFS")

    plt.subplot(2, 3, 2)
    plt.imshow(before_psf)
    plt.colorbar()
    plt.title("PSF - no WFS")

    plt.subplot(2, 3, 3)
    plt.title("Aberrations")
    plt.imshow(aberrations.read())
    plt.colorbar()

    alg.wavefront = WFSController.State.OPTIMIZED
    after = mic.read()
    after_psf = mic._psf

    plt.subplot(2, 3, 4)
    plt.imshow(after)
    plt.title("Image - WFS")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(after_psf)
    plt.colorbar()
    plt.title("PSF - WFS")

    plt.subplot(2, 3, 6)
    plt.imshow(-slm.phases.read())
    plt.colorbar()
    plt.title("Corrections × -1")
    plt.show(block=True)
