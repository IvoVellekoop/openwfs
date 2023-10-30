from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi, SingleRoiSquare, SelectRoiSquare, SelectRoiCircle
from openwfs.slm import SLM
from test_functions import calculate_enhancement,make_angled_wavefront, angular_difference, measure_feedback, plot_dense_grid, plot_dense_grid_no_empty_spaces
import matplotlib.pyplot as plt
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
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
#    ideal_wf = (np.load("..//..//WFS_experiments//16_06_2023 Headless wfs experiment//fourier4//optimised_wf.npy")/255)*2*np.pi - np.pi
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
    sim.set_data(ideal_wf)
    sim.trigger()

    controller = Controller(detector=roi_detector, slm=sim)
    alg = BasicFDR(k_angles_min=-1, k_angles_max=1, phase_steps=30, overlap=0.1, controller=controller)
    t = alg.execute()
    optimised_wf = np.angle(t)

    plt.figure()
    plt.imshow(ideal_wf)
    plt.colorbar(label='Phase offset (radians)')

    plt.figure()
    plt.imshow(optimised_wf)
    plt.colorbar(label='Phase offset (radians)')
    plt.show()

    enhancement = calculate_enhancement(sim, optimised_wf, x=256, y=256)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf, x=256, y=256)
    print(f'Enhancement for 25 Fourier modes is {(enhancement/enhancement_perfect)*100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True

import time
def enhancement_ssa():
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
    # s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=2, n_y=2, phase_steps=30, controller=controller)
    t = alg.execute()


    optimised_wf = np.angle(t)

    plt.figure()
    plt.imshow(ideal_wf)
    plt.colorbar(label='Phase offset (radians)')

    plt.figure()
    plt.imshow(optimised_wf)
    plt.colorbar(label='Phase offset (radians)')
    plt.show()


    enhancement = calculate_enhancement(sim, optimised_wf)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf)
    print(f'Enhancement for 25 SSA modes is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"SSA algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}")
    else:
        return True

def enhancement_characterising_fourier():
    sim = SimulatedWFS(width=512, height=512,beam_profile_fwhm=500)

    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    roi_detector.trigger()

#    correct_wf = (np.load("..//..//WFS_experiments//16_06_2023 Headless wfs experiment//fourier4//optimised_wf.npy")/255)*2*np.pi - np.pi
#    correct_wf = make_angled_wavefront(512, -4, 3)
    correct_wf = (data.camera()/255)*2*np.pi-np.pi
    # correct_wf = (np.fliplr(data.camera()) / 255) * 2 * np.pi - np.pi
#    correct_wf = np.zeros((1056,1056))
    sim.set_ideal_wf(correct_wf)

#    s1 = SLM(left=0, width=300, height=300)
    intermediate = True
    controller = Controller(detector=roi_detector, slm=sim)
    alg = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=30, high_modes=0,high_phase_steps=17, intermediates=intermediate, controller=controller)
    t = alg.execute()

    print(alg.t_left)

    print(alg.k_left)
    plt.figure()
    plt.scatter(alg.k_left[0, :], alg.k_left[1, :], c=abs(alg.t_left), marker='s', cmap='viridis', s=400, edgecolors='k')
    plt.colorbar(label='t_abs')


    plt.figure()
    plt.scatter(alg.k_right[0, :], alg.k_right[1, :], c=abs(alg.t_right), marker='s', cmap='viridis', s=400, edgecolors='k')
    plt.colorbar(label='t_abs')


    optimised_wf = np.angle(t)

    plt.figure()
    plt.title('Correct wavefront')
    plt.imshow(correct_wf, cmap='hsv')
    plt.colorbar()
    plt.clim([-np.pi,np.pi])

    plt.figure()
    plt.title('Optimised wavefront')
    plt.imshow(optimised_wf, cmap='hsv')
    plt.colorbar()
    plt.clim([-np.pi, np.pi])
    plt.figure()
    plt.title('Angular difference')
    plt.imshow(angular_difference(optimised_wf,correct_wf), cmap='hsv')
    plt.colorbar()
    plt.clim([-np.pi, np.pi])

    plt.show()

    predicted_enhancement = [0]
    previous= 0
    decrease_ind = []
    ncount = 0
    modenumbers = [0]
    combined_t = np.append(alg.t_left,alg.t_right)
    measured_modes = alg.added_modes[1:]

    for n,modes in enumerate(alg.added_modes[1:]):
        nmodes = len(modes)
        if nmodes == 8:
            nmodes = 9
        predicted_enhancement.append(
            predicted_enhancement[-1] + np.sum(abs(combined_t[ncount:(ncount + nmodes)])))
        ncount += nmodes
        modenumbers.append(ncount)

    # Create the first plot and axis
    fig, ax1 = plt.subplots()

    if intermediate:

        # Plot the first dataset
        line1, = ax1.plot(modenumbers,alg.intermediate_enhancements, 'b.', label='intermediate enhancement')
        ax1.set_xlabel('Number of modes')
        ax1.set_ylabel('intermediate enhancement', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        # Create a second y-axis for the same x-axis
        ax2 = ax1.twinx()

        # Plot the second dataset
        line2, = ax2.plot(modenumbers,np.sqrt(predicted_enhancement), 'r.', label='Cumulative signal strength sqrt(abs(t))')
        ax2.set_ylabel('Cumulative signal strength sqrt(abs(t))', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')


        # Combine legends from both axes
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=0)

        plt.title('Simulation: Actual enhancement vs cumulative mode strength')


    # making the added-modes-by-added-modes plots

    # for n,t in enumerate(alg.intermediate_t):
    #     plt.figure()
    #     plt.title(f'Calculated WF {n}')
    #     plt.imshow(np.angle(t))
        # plt.figure()
        # plt.title(f"angular difference {n}")
        # plt.imshow(angular_difference(np.angle(t),correct_wf))

    plt.figure()
    plt.plot(modenumbers,alg.intermediate_enhancements,'.')
    plt.show()
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
    sim.read = data.camera # overriding the read function for more meaningful images

    detector = SelectRoiCircle(sim)
    detector.trigger()
    print(detector.read())
    plt.imshow(detector.read_circle())
    plt.show()

    detector = SelectRoiSquare(sim)
    detector.trigger()
    print(detector.read())
    plt.imshow(detector.read_square())
    plt.show()
    return True

# print(flat_wf_response_ssa())
# print(flat_wf_response_fourier())
# print(enhancement_fourier())
# print(enhancement_ssa())
# print(enhancement_characterising_fourier())
# print(square_selection_detector_test())
# print(drawing_detector())