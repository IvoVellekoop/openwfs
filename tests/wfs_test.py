from simulation.simulation import SimulatedWFS
from wfs_functions import WfsExperiment
from fourier import FourierDualRef
from ssa import StepwiseSequential
import numpy as np
from skimage import data
from algorithms import StepwiseSequential2
from feedback import SimpleCameraFeedback

"""This """
def calculate_enhancement(simulation,wfs_experiment):
    simulation.set_data(0)
    simulation.trigger()
    simulation.wait()
    feedback_before = sum(simulation.get_image()[250:251,250:251])

    simulation.set_data(wfs_experiment.optimised_wf)
    simulation.trigger()
    simulation.wait()
    feedback_after = sum(simulation.get_image()[250:251,250:251])

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
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500])) # correct wf = flat
    sim.E_input_slm = np.ones([500, 500]) # set image plane size, and gauss off

    wfs.algorithm = StepwiseSequential()
    wfs.ranges = [[250, 251], [250, 251]] # get feedback from the center of the image plane
    wfs.algorithm.n_slm_fields = 3

    wfs.slm_object = sim
    wfs.camera_object = sim

    wfs.execute = 1
    if np.std(wfs.optimised_wf) > 0:
        raise Exception("Response flat wavefront not flat")
    else:
        return True

def flat_wf_response_ssa2():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500])) # correct wf = flat
    sim.E_input_slm = np.ones([500, 500]) # set image plane size, and gauss off

    f = SimpleCameraFeedback(camera=sim, slm=sim, roi_x=250, roi_y=250, roi_radius=1)
    alg = StepwiseSequential2(Nx=8, Ny=4, feedback=f, slm=sim)

    t = alg.execute()
    optimised_wf = np.angle(t)

    if np.std(optimised_wf) > 0:
        raise Exception("Response flat wavefront not flat")
    else:
        return True

def wf_response_fourier():

    wfs = WfsExperiment()
    sim = SimulatedWFS()
    sim.set_ideal_wf(data.camera()) # correct wf = flat
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

    if calculate_enhancement(sim,wfs) < 3:
        raise Exception("Fourier algorithm does not enhance focus as much as expected")
    else:
        return True


print(flat_wf_response_ssa2())
print(flat_wf_response_fourier())
print(flat_wf_response_ssa())
print(wf_response_fourier())
