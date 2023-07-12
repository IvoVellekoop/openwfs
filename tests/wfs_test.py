from simulation.simulation import SimulatedWFS
from wfs_functions import WfsExperiment
from fourier import FourierDualRef
from ssa import SSA
import numpy as np
from skimage import data

"""This """
def calculate_enhancement(simulation,wfs_experiment):
    simulation.set_data(0)
    simulation.trigger()
    simulation.wait()
    feedback_before = float(simulation.get_image()[250:251,250:251])

    simulation.set_data(wfs_experiment.optimised_wf)
    simulation.trigger()
    simulation.wait()
    feedback_after = float(simulation.get_image()[250:251,250:251])

    return feedback_after/feedback_before

def flat_wf_response_fourier():

    wfs = WfsExperiment()
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500])) # correct wf = flat
    sim.E_input_slm = np.ones([500, 500]) # set image plane size, and gauss off

    wfs.algorithm = FourierDualRef()
    wfs.ranges = [[250, 251], [250, 251]] # get feedback from the center of the image plane
    wfs.algorithm.ky_angles_max = 1
    wfs.algorithm.ky_angles_min = -1
    wfs.algorithm.kx_angles_max = 1
    wfs.algorithm.kx_angles_min = -1
    wfs.algorithm.kx_angles_stepsize = 1
    wfs.algorithm.ky_angles_stepsize = 1

    wfs.algorithm.build_kspace()
    wfs.slm_object = sim
    wfs.camera_object = sim

    wfs.execute = 1
    if np.std(wfs.optimised_wf) > 0:
        raise Exception("Response flat wavefront not flat")
    else:
        return True


def flat_wf_response_ssa():

wfs = WfsExperiment()

wfs.algorithm = FourierDualRef()
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)

wfs.algorithm.kx_angles_stepsize = 2
wfs.algorithm.ky_angles_stepsize = 2
wfs.algorithm.ky_angles_max = 0
wfs.algorithm.build_kspace()
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)

wfs.algorithm.set_kspace([-4, 2], [7, 9])
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)
wfs.slm_object = SimulatedWFS()
wfs.camera_object = SimulatedWFS()

wfs.execute = 1
# or you can use wfs.on_execute(), works either way