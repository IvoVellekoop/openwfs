import matplotlib.pyplot as plt

import set_path
import numpy as np
from openwfs.algorithms import BasicFDR
from openwfs.algorithms.utilities import WFSController
from openwfs.processors import SingleRoi
from openwfs.simulation import Microscope, MockSource, MockSLM, SimulatedWFS
from openwfs.utilities import grab_and_show, imshow
import skimage
import astropy.units as u

numerical_aperture = 1.0
aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)

img = np.zeros((1000, 1000), dtype=np.int16)
signal_location = (256, 256)
img[signal_location] = 100

src = MockSource(img, 400 * u.nm)

slm = MockSLM(shape=(1000, 1000))

sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=numerical_aperture,
                 aberrations=aberration,
                 wavelength=800 * u.nm)

cam = sim.get_camera(analog_max=100)
roi_detector = SingleRoi(cam, x=256, y=256, radius=0)  # Only measure that specific point
alg = BasicFDR(feedback=roi_detector, slm=slm, slm_shape=(1000, 1000), k_angles_min=-3, k_angles_max=3, phase_steps=3)
controller = WFSController(alg)
f = roi_detector.read()

devices = {
    'cam': cam,
    'wfs_controller': controller,
    'slm': slm,
    'stage': sim.xy_stage,
    'microscope': sim,
    'wfs': alg}

controller.wavefront = WFSController.State.FLAT_WAVEFRONT
before = roi_detector.read()
controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
after = roi_detector.read()
imshow(controller._optimized_wavefront)
print(after / before)
