from simulation.simulation import SimulatedWFS
import numpy as np
from skimage import data
from algorithms import StepwiseSequential, BasicFDR
from feedback import Controller, SingleRoi
from test_functions import calculate_enhancement,make_angled_wavefront
from slm import SLM
import matplotlib.pyplot as plt

def flat_wf_response_fourier():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    roi_detector = SingleRoi(sim, x=250, y=250, radius=2)
    roi_detector.trigger() #
    controller = Controller(detector=roi_detector, slm=sim)

    alg = BasicFDR(k_angles_min=-1, k_angles_max=1, phase_steps=3, overlap=0.1, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    if np.std(optimised_wf) > 0.001:
        raise Exception("Response flat wavefront not flat")
    else:
        return True



def flat_wf_response_ssa():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
    roi_detector = SingleRoi(sim, x=250, y=250, radius=1)

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
    roi_detector = SingleRoi(sim, x=250, y=250, radius=1)
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
#    s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = BasicFDR(k_angles_min=-2, k_angles_max=2, phase_steps=3, overlap=0.1, controller=controller)
    t = alg.execute()
    optimised_wf = (np.angle(t))

    enhancement = calculate_enhancement(sim, optimised_wf)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf)
    print(f'Enhancement is {(enhancement/enhancement_perfect)*100} % of possible enhancement')

    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True


def enhancement_ssa():
    sim = SimulatedWFS(beam_profile_fwhm=500)
    roi_detector = SingleRoi(sim, x=250, y=250, radius=1)

    sim.set_ideal_wf(make_angled_wavefront(500,-1,1))
#    s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=3, n_y=3, phase_steps=5, controller=controller)
    t = alg.execute()


    optimised_wf = np.angle(t)


    enhancement = calculate_enhancement(sim, optimised_wf)
    enhancement_perfect = calculate_enhancement(sim, make_angled_wavefront(500,-1,1))
    print(f'Enhancement is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')

    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}")
    else:
        return True

print(flat_wf_response_ssa())
print(flat_wf_response_fourier())
print(enhancement_fourier())
print(enhancement_ssa())