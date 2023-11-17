import matplotlib.pyplot as plt

import set_path
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.processors import SingleRoi
from openwfs.simulation import SimulatedWFS,Microscope,MockCamera,MockSource,MockXYStage,MockSLM
import skimage
from openwfs.slm import SLM
from openwfs.slm.patterns import tilt,disk
import astropy.units as u

aberrations = skimage.data.camera() * ((2*np.pi) / 255.0)
stage = MockXYStage(0.1 * u.um, 0.1 * u.um)
img = np.zeros((1000, 1000), dtype=np.int16)
img[256,256] = 100
src = MockCamera(MockSource(img, 450 * u.nm))

slm = MockSLM(shape = (512,512))

aberration = MockSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)

sim = Microscope(source=src, slm=slm.pixels(), magnification=45,numerical_aperture=0.8, aberrations=aberration,xy_stage=stage,wavelength=800*u.nm)

devices = {
    'cam': sim.camera,
    'stage': sim.xy_stage,
    'microscope': sim}




