import numpy as np
import astropy.units as u
from openwfs.processors import SingleRoi
from openwfs.simulation import SLM, Microscope, Shutter
from openwfs.algorithms import StepwiseSequential, troubleshoot
from openwfs.utilities import set_pixel_size

# === Define virtual devices for a WFS simulation ===
# Note: these virtual-device classes may be replaced
# with real-device classes to run a real WFS experiment

# Define aberration as a pattern of random phases at the pupil plane
aberrations = np.random.uniform(size=(40, 40)) * 2 * np.pi

# Define specimen as an image with several bright pixels
specimen = np.zeros((120, 120))
specimen[60, 60] = 1e4
specimen = set_pixel_size(specimen, pixel_size=200 * u.nm)

# Simulate an SLM with incorrect phase response
# Also simulate a shutter that can turn off the light
# The SLM is conjugated to the back pupil plane
slm = SLM(shape=(100, 100),
          phase_response=(np.arange(256) / 128 * np.pi) * 1.4)
shutter = Shutter(slm.field)

# Simulate a WFS microscope looking at the specimen
sim = Microscope(source=specimen, incident_field=shutter, aberrations=aberrations, wavelength=800 * u.nm)

# Simulate a camera device with gaussian noise and shot noise
cam = sim.get_camera(analog_max=1e4, shot_noise=True, gaussian_noise_std=0.4)

# Define feedback as circular region of interest in the center of the frame
roi_detector = SingleRoi(cam, radius=0.1)

# === Run wavefront shaping experiment ===
# Use the stepwise sequential (SSA) WFS algorithm
alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=10, n_y=10, phase_steps=16)

# Define a region of interest to determine background intensity
roi_background = SingleRoi(cam, radius=8)

# Run WFS troubleshooter and output a report to the console
trouble = troubleshoot(algorithm=alg, background_feedback=roi_background,
                       frame_source=cam, shutter=shutter, do_stability_test=False)
trouble.report()
