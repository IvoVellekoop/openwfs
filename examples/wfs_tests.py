from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi, SingleRoiSquare, SelectRoiSquare, SelectRoiCircle
from test_functions import calculate_enhancement,make_angled_wavefront, angular_difference, measure_feedback, plot_dense_grid, plot_dense_grid_no_empty_spaces
import matplotlib.pyplot as plt
from openwfs.slm import SLM
import matplotlib
import astropy.units as u
from skimage import data
from time import sleep

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
    # s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = BasicFDR(k_angles_min=-2, k_angles_max=2, phase_steps=3, overlap=0.1, controller=controller)
    t = alg.execute()
    optimised_wf = np.angle(t)

    plt.figure()
    plt.imshow(ideal_wf)

    plt.figure()
    plt.imshow(optimised_wf)
    plt.show()

    enhancement = calculate_enhancement(sim, optimised_wf, x=256, y=256)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf, x=256, y=256)
    print(f'Enhancement for 25 Fourier modes is {(enhancement/enhancement_perfect)*100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True


def enhancement_ssa():
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera()/255)*2*np.pi
    sim.set_ideal_wf(ideal_wf)
    # s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

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
    intermediate = False
    controller = Controller(detector=roi_detector, slm=sim)
    alg = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=100, high_modes=0,high_phase_steps=17, intermediates=intermediate, controller=controller)
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

def fourier_basic_pathfinding_comparison():
    sim = SimulatedWFS(width=512, height=512, beam_profile_fwhm=300)

    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    roi_detector.trigger()
    #correct_wf = make_angled_wavefront(512, -5, 4)
    # correct_wf = (data.camera()/255)*2*np.pi-np.pi
    correct_wf = (np.load("..//..//WFS_experiments//16_06_2023 Headless wfs experiment//fourier2//optimised_wf.npy") / 255) * 2 * np.pi - np.pi
    sim.set_ideal_wf(correct_wf)


    plt.imshow((correct_wf%(2*np.pi))-np.pi)
    plt.colorbar(label='Phase offset (radians)')
    basic_n = 5
    matplotlib.rcParams.update({'font.size': 16})


    controller = Controller(detector=roi_detector, slm=sim)
    alg_char = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=(basic_n*2+1)**2+5, high_modes=0, high_phase_steps=17,
                            intermediates=False, controller=controller)


    alg_basic= BasicFDR(k_angles_min=-basic_n, k_angles_max=basic_n, phase_steps=3, overlap=0.1, controller=controller)

    alg_char.execute()
    alg_basic.execute()

    t_left = alg_char.t_left
    t_right = alg_char.t_right
    k_left = alg_char.k_left
    k_right = alg_char.k_right


    t_basic_left = alg_basic.t_left
    t_basic_right = alg_basic.t_right
    basic_enhancements = []
    pathfinding_enhancements = []
    modes = []
    for n in range(1,basic_n+1):

        tleft = t_basic_left[basic_n-n:basic_n+n+1,basic_n-n:basic_n+n+1]
        tright = t_basic_right[basic_n-n:basic_n+n+1,basic_n-n:basic_n+n+1]
        t_cropped = np.append(tleft,tright)
        n_angles = (n*2+1)**2
        modes.append(n_angles)
        alg_basic.k_angles_min = -n
        alg_basic.k_angles_max = n
        alg_basic.build_kspace()


        t = alg_basic.compute_t([[t] for t in t_cropped])
        x, y = np.meshgrid(np.arange(-n,n+1), np.arange(-n,n+1))

        plot_dense_grid_no_empty_spaces(x.flatten(),y.flatten(),abs(tleft.flatten()))
        plt.xlim([-8.5, 8.5])
        plt.ylim([-8.5, 8.5])
        plt.figure()
        plt.imshow(np.angle(t))
        plt.title(f'Basic fourier {n_angles} Modes')
        basic_enhancements.append(measure_feedback(sim, np.angle(t)))

        t_pathfinding = alg_char.compute_t(t_left[:n_angles],t_right[:n_angles],k_left[:n_angles],k_right[:n_angles])
        if n_angles==9:
            plot_dense_grid_no_empty_spaces(k_left[0, :n_angles], k_left[1, :n_angles], t_left[:n_angles])
        else:
            plot_dense_grid(k_left[0,:n_angles], k_left[1,:n_angles], t_left[:n_angles])
        plt.xlim([-8.5, 8.5])
        plt.ylim([-8.5, 8.5])
        plt.figure()
        pathfinding_enhancements.append(measure_feedback(sim, np.angle(t_pathfinding)))
        plt.imshow(np.angle(t_pathfinding))
        plt.title(f'Pathfinding fourier {n_angles} Modes')
    plt.figure()
    plt.plot(modes,basic_enhancements,'r.')
    plt.figure(figsize=(10, 8))
    plt.plot(modes,pathfinding_enhancements,'b.')
    plt.xlabel('Number of modes')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(['Basic fourier','Pathfinding fourier'],loc='upper left')
    plt.show()
    return True
# print(flat_wf_response_ssa())
# print(flat_wf_response_fourier())
# print(enhancement_fourier())
# print(enhancement_ssa())
# print(enhancement_characterising_fourier())
# print(square_selection_detector_test())
# print(drawing_detector())
print(fourier_basic_pathfinding_comparison())

