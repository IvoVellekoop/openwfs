import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from Simulation.Simulation import SimulatedWFS, make_gaussian
from Fourier import FourierDualRef
from WFS_functions import WfsExperiment
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\hardware\\generic_binding')
from SLMwrapper import SLM, set_circular_geometry

# initialisation
sim = SimulatedWFS()
wfs = WfsExperiment()
fourier = FourierDualRef()
wfs.algorithm = fourier
wfs.slm_object = sim
wfs.camera_object = sim


# set the simulation details
sim.set_ideal_wf(data.camera())
sim.E_input_slm = make_gaussian(500,fwhm=1000)

# set the algorithm details
k_xset = np.arange(-2,2+1,1)
k_yset = np.arange(-2,2+1,1)
fourier.set_kspace(k_xset,k_yset)

wfs.execute = 1
plt.imshow(wfs.optimised_wf)
plt.show()