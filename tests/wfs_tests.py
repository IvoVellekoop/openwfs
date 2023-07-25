from simulation.simulation import SimulatedWFS
import numpy as np
from skimage import data
from algorithms import StepwiseSequential, FourierDualRef
from feedback import Controller, SingleRoi
from test_functions import calculate_enhancement,make_angled_wavefront
import matplotlib.pyplot as plt

def flat_wf_response_fourier():
    sim = SimulatedWFS(active_plotting=True)
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
    sim.E_input_slm = np.ones([500, 500])  # set image plane size, and gauss off
    roi_detector = SingleRoi(sim, x=250, y=250, radius=2)
    roi_detector.trigger() #
    controller = Controller(detector=roi_detector, slm=sim)

    alg = FourierDualRef(k_angles_min=-1, k_angles_max=1, phase_steps=3, overlap=0.1, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    if np.std(optimised_wf) > 0.001:
        raise Exception("Response flat wavefront not flat")
    else:
        return True



def flat_wf_response_ssa():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
    sim.E_input_slm = np.ones([500, 500])  # set image plane size, and gauss off

    roi_detector = SingleRoi(sim, x=250, y=250, radius=1)
    roi_detector.trigger()
    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(N_x=3, N_y=3, phase_steps=3, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    if np.std(optimised_wf) > 0.001:
        raise Exception("Response flat wavefront not flat")
    else:
        return True


def enhancement_fourier():
    sim = SimulatedWFS()
    sim.E_input_slm = np.ones([500, 500])  # set image plane size, and gauss off
    roi_detector = SingleRoi(sim, x=250, y=250, radius=1)
    roi_detector.trigger()

    sim.set_ideal_wf(data.camera())   # correct wf = flat
#    s1 = SLM(0, left=0, width=300, height=300)

    controller = Controller(detector=roi_detector, slm=sim)
    alg = FourierDualRef(k_angles_min=-2, k_angles_max=2, phase_steps=3, overlap=0.1, controller=controller)

    t = alg.execute()
    optimised_wf = (np.angle(t)+np.pi)/((np.pi*2)/255)
    enhancement = calculate_enhancement(sim, optimised_wf)
    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True



print(flat_wf_response_ssa())
print(flat_wf_response_fourier())
print(enhancement_fourier())