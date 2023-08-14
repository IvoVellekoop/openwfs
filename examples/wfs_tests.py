from openwfs.openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.openwfs.feedback import Controller, SingleRoi, SingleRoiSquare, SelectRoiSquare
from test_functions import calculate_enhancement,make_angled_wavefront, angular_difference
import matplotlib.pyplot as plt
from openwfs.openwfs.slm import SLM
import astropy.units as u
from skimage import data


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
    alg = StepwiseSequential(n_x=6, n_y=6, phase_steps=3, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    if np.std(optimised_wf) > 0.001:
        raise Exception("Response flat wavefront not flat")
    else:
        return True


def enhancement_fourier():
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=1)
#    ideal_wf = (np.load("..//..//WFS_experiments//16_06_2023 Headless wfs experiment//fourier4//optimised_wf.npy")/255)*2*np.pi - np.pi
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
    sim.set_data(ideal_wf)
    sim.trigger()
#    s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = BasicFDR(k_angles_min=-2, k_angles_max=2, phase_steps=3, overlap=0.1, controller=controller)
    t = alg.execute()
    optimised_wf = np.angle(t)
    plt.figure()

    enhancement = calculate_enhancement(sim, optimised_wf, x=256, y=256)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf, x=256, y=256)
    print(f'Enhancement for 25 Fourier modes is {(enhancement/enhancement_perfect)*100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True


def enhancement_ssa():
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=1)
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
#    s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=5, n_y=5, phase_steps=5, controller=controller)
    t = alg.execute()


    optimised_wf = np.angle(t)


    enhancement = calculate_enhancement(sim, optimised_wf)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf)
    print(f'Enhancement for 25 SSA modes is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"SSA algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}")
    else:
        return True

def enhancement_characterising_fourier():
    sim = SimulatedWFS(width=1056, height=1056)
    roi_detector = SingleRoi(sim, x=528, y=528, radius=1)
    roi_detector.trigger()

#    correct_wf = (np.load("..//..//WFS_experiments//16_06_2023 Headless wfs experiment//fourier4//optimised_wf.npy")/255)*2*np.pi - np.pi
    correct_wf = make_angled_wavefront(1056, -2, 1)
#    correct_wf = np.zeros((1056,1056))
    sim.set_ideal_wf(correct_wf)

#    s1 = SLM(left=0, width=300, height=300)

    controller = Controller(detector=roi_detector, slm=sim)
    alg = CharacterisingFDR(phase_steps=3, overlap=0.2, max_modes=20, controller=controller)
    t = alg.execute()

    optimised_wf = np.angle(t)
    plt.figure()
    plt.imshow(optimised_wf)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(angular_difference(optimised_wf,correct_wf))
    plt.colorbar()
    plt.show()
    enhancement_perfect = calculate_enhancement(sim, correct_wf, x=528, y=528)
    enhancement = calculate_enhancement(sim, optimised_wf, x=528, y=528)

    print(f'Enhancement is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')
    print(enhancement)
    print(enhancement_perfect)
    if enhancement < 3:
        raise Exception(
            f"Fourier algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}")
    else:
        return True

def square_selection_detector_test():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    width = 20
    height = 20
    detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
    detector.trigger() #
    if sim.read()[250,250]/(width*height) != detector.read():
        raise Exception(f"Square detector not working as expected")
    return True


def drawing_detector():

    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    detector = SelectRoiSquare(sim)
    detector.trigger() #
    print(detector.read())
    plt.imshow(detector.read_square())
    plt.show()
    return True

# print(flat_wf_response_ssa())
# print(flat_wf_response_fourier())
# print(enhancement_fourier())
# print(enhancement_ssa())
# print(enhancement_characterising_fourier())
print(square_selection_detector_test())
print(drawing_detector())