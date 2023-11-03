from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi
import sys
sys.path.append('..//')
import matplotlib.pyplot as plt
from skimage import data

def enhancement_characterising_fourier():
    """
    Test the performance of the pathfinding Fourier-based algorithm.
    Using the functionality in 'CharacterisingFDR', we can measure the enhancement during the experiment.

    We can also predict enhancement, using the measured mode strength. We show the results.
    """
    sim = SimulatedWFS(width=512, height=512, beam_profile_waist=1.5)

    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    roi_detector.trigger()

    correct_wf = (data.camera() / 255) * 2 * np.pi - np.pi
    sim.set_ideal_wf(correct_wf)

    intermediate = True
    controller = Controller(detector=roi_detector, slm=sim)
    alg = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=40, high_modes=0, high_phase_steps=17,
                            intermediates=intermediate, controller=controller)
    t = alg.execute()

    print(alg.t_left)

    print(alg.k_left)
    plt.figure()
    k = alg.k_left
    plt.imshow(abs(alg.get_dense_matrix(alg.k_left,alg.t_left)),extent=(min(k[0,:])-0.5,max(k[0,:])+0.5,min(k[1,:])-0.5,max(k[1,:])+0.5))
    plt.colorbar(label='t_abs')

    plt.figure()
    k = alg.k_right
    plt.imshow(abs(alg.get_dense_matrix(alg.k_right,alg.t_right)),extent=(min(k[0,:])-0.5,max(k[0,:])+0.5,min(-k[1,:])-0.5,max(-k[1,:])+0.5))
    plt.colorbar(label='t_abs')
    print(min(alg.k_right[0,:]))
    print(max(alg.k_right[0, :]))
    print(min(alg.k_right[1,:]))
    print(max(alg.k_right[1, :]))
    optimised_wf = np.angle(t)

    plt.figure()
    plt.title('Correct wavefront')
    plt.imshow(correct_wf, cmap='hsv')
    plt.colorbar()
    plt.clim([-np.pi, np.pi])

    plt.figure()
    plt.title('Optimised wavefront')
    plt.imshow(optimised_wf, cmap='hsv')
    plt.colorbar()
    plt.clim([-np.pi, np.pi])


    predicted_enhancement = [0]

    ncount = 0
    modenumbers = [0]
    combined_t = np.append(alg.t_left, alg.t_right)


    for n, modes in enumerate(alg.added_modes[1:]):
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
        line1, = ax1.plot(modenumbers, alg.intermediate_enhancements, 'b.', label='intermediate enhancement')
        ax1.set_xlabel('Number of modes')
        ax1.set_ylabel('intermediate enhancement', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        # Create a second y-axis for the same x-axis
        ax2 = ax1.twinx()

        # Plot the second dataset
        line2, = ax2.plot(modenumbers, np.sqrt(predicted_enhancement), 'r.',
                          label='Cumulative signal strength sqrt(abs(t))')
        ax2.set_ylabel('Cumulative signal strength sqrt(abs(t))', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=0)

        plt.title('Simulation: Actual enhancement vs cumulative mode strength')

    plt.show()
    return True

if __name__=="__main__":
    print(enhancement_characterising_fourier())