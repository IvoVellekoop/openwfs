import matplotlib.pyplot as plt

import set_path
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.devices.wfs_device import WFSController
from openwfs.processors import SingleRoi
from openwfs.simulation import SimulatedWFS,Microscope,MockCamera,MockSource,MockXYStage,MockSLM
import skimage

from openwfs.slm import SLM
from openwfs.slm.patterns import tilt,disk
import astropy.units as u

aberrations = skimage.data.camera() * ((2*np.pi) / 255.0)+np.pi

aberration = MockSource(aberrations, pixel_size=1.0 / (512) * u.dimensionless_unscaled)

img = np.zeros((1000, 1000), dtype=np.int16)
img[256,256] = 100

img = np.zeros((1000, 1000), dtype=np.int16)
signal_location = (256,256)

img[signal_location] = 100
src = MockSource(img, 400 * u.nm)

slm = MockSLM(shape=(1000,1000))

sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=1, aberrations=aberration,
                 wavelength=800 * u.nm,camera_pixel_size=400 * u.nm,
                 camera_resolution=(1000,1000),analog_max=100)

roi_detector = SingleRoi(sim.camera, x=256, y=256, radius=0) # Only measure that specific point

# alg = BasicFDR(feedback=roi_detector,slm=slm,slm_shape=(1000,1000),k_angles_min=-1,k_angles_max=1,phase_steps=3)
alg = StepwiseSequential(feedback=roi_detector,slm=slm)
controller = WFSController(alg)

controller.execute_button = True
alg._slm = SLM(0)
controller.show_optimized_wavefront = True

import time
time.sleep(5)

devices = {
    'cam': sim.camera,
    'wfs_controller': controller,
    'slm': slm,
    'stage': sim.xy_stage,
    'microscope': sim,
    'wfs': alg}




