import numpy as np
import astropy.units as u
from openwfs.processors import SingleRoi
from openwfs.simulation import MockSource, MockSLM, Microscope
from openwfs.algorithms import StepwiseSequential, troubleshoot
from openwfs.utilities import set_pixel_size


# === Define virtual devices for a WFS simulation ===
# Note: these virtual-device classes may be replaced
# with real-device classes to run a real WFS experiment

# Define aberration as a pattern of random phases at the pupil plane
aberrations = np.random.uniform(size=(100, 100)) * 2 * np.pi

# Define specimen as an imge with several bright pixels
specimen = np.zeros((120, 120))
specimen[60, (60, 70, 80, 90, 100, 110)] = 1000
specimen = set_pixel_size(specimen, pixel_size=200*u.nm)

# Simulate an SLM with incorrect phase response
slm = MockSLM(shape=(100, 100), phase_response=(np.arange(256)/128 * np.pi) ** 1.2)

# Simulate a WFS microscope looking at the specimen with a specified wavelength
sim = Microscope(source=specimen, slm=slm, aberrations=aberrations, wavelength=800*u.nm)

# Get a camera and feedback device object
cam = sim.get_camera(shot_noise=True)
roi_detector = SingleRoi(cam, radius=1)


# === Run wavefront shaping experiment ===
# Define the WFS stepwise sequential (SSA) algorithm
alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=10, n_y=10, phase_steps=8)

# Run WFS troubleshooter and output a report to the console
trouble = troubleshoot(algorithm=alg, frame_source=cam, shutter=cam.shutter, do_stability_test=False)
trouble.report()
