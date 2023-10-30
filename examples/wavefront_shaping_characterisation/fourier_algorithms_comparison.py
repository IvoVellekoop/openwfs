from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('..//')
from test_functions import calculate_enhancement,make_angled_wavefront, angular_difference, measure_feedback, plot_dense_grid, plot_dense_grid_no_empty_spaces



def fourier_basic_pathfinding_comparison():
    '''
    Runs a wavefront shaping simulation using both the basic Fourier & the pathfinding Fourier algorithms,
    It also plots the results of the experiment with an increasing amount of modes for each method.
    '''
    sim = SimulatedWFS(width=512, height=512, beam_profile_fwhm=300)

    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    roi_detector.trigger()

    correct_wf = (np.load('../data/fourier/optimised_wf.npy') / 255) * 2 * np.pi - np.pi
    sim.set_ideal_wf(correct_wf)


    plt.imshow((correct_wf%(2*np.pi))-np.pi)
    plt.colorbar(label='Phase offset (radians)')
    basic_n = 4
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
    k_basic_left = alg_basic.k_left
    k_basic_right = alg_basic.k_right

    basic_enhancements = []
    pathfinding_enhancements = []
    modes = []
    for n in range(1,basic_n+1):
        n_angles = (n*2+1)**2
        modes.append(n_angles)

        # select the subset of the k-space, so
        x_range = np.logical_and(k_basic_left[0] >= -n, k_basic_left[0] <= n)
        y_range = np.logical_and(k_basic_left[1] >= -n, k_basic_left[1] <= n)
        combined_range = np.logical_and(x_range, y_range)
        plot_dense_grid_no_empty_spaces(k_basic_left[0,combined_range], k_basic_left[1, combined_range], t_basic_left[combined_range])
        plt.xlim([-8.5, 8.5])
        plt.ylim([-8.5, 8.5])

        t_basic = alg_basic.compute_t(t_basic_left[combined_range], t_basic_right[combined_range], k_basic_left[:,combined_range], k_basic_right[:,combined_range])
        # plt.figure()
        # plt.imshow(np.angle(t_basic))
        # plt.title(f'Basic fourier {n_angles} Modes')
        basic_enhancements.append(measure_feedback(sim, np.angle(t_basic)))

        t_pathfinding = alg_char.compute_t(t_left[:n_angles],t_right[:n_angles],k_left[:n_angles],k_right[:n_angles])
        if n_angles==9:
            plot_dense_grid_no_empty_spaces(k_left[0, :n_angles], k_left[1, :n_angles], t_left[:n_angles])
        else:
            plot_dense_grid(k_left[0,:n_angles], k_left[1,:n_angles], t_left[:n_angles])
        plt.xlim([-8.5, 8.5])
        plt.ylim([-8.5, 8.5])
        # plt.figure()
        pathfinding_enhancements.append(measure_feedback(sim, np.angle(t_pathfinding)))
        # plt.imshow(np.angle(t_pathfinding))
        # plt.title(f'Pathfinding fourier {n_angles} Modes')
    plt.figure()
    plt.plot(modes,basic_enhancements,'r.')
    plt.plot(modes,pathfinding_enhancements,'b.')
    plt.xlabel('Number of modes')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(['Basic fourier','Pathfinding fourier'],loc='upper left')
    plt.show()
    return True

if __name__ == '__main__':
    print(fourier_basic_pathfinding_comparison())