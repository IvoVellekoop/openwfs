import set_path
from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi, SingleRoiSquare, SelectRoiSquare, SelectRoiCircle
from openwfs.slm import SLM
from functions import calculate_enhancement, make_angled_wavefront, angular_difference, measure_feedback, \
    plot_dense_grid, plot_dense_grid_no_empty_spaces
import matplotlib.pyplot as plt
from skimage import data


def test_flat_wf_response_fourier():
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.

    test the optimised wavefront by checking if it has irregularities.
    Since a flat wavefront at 0 or pi have the same effect, the absolute value of the front is irrelevant.
    """
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    roi_detector = SingleRoi(sim, x=250, y=250, radius=2)
    roi_detector.trigger()  #
    controller = Controller(detector=roi_detector, slm=sim)

    alg = BasicFDR(k_angles_min=-1, k_angles_max=1, phase_steps=3, overlap=0.1, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    # test the optimised wavefront by checking if it has irregularities. Since a flat wavefront at 0 or pi
    assert np.std(optimised_wf) < 0.001  # "Response flat wavefront not flat"


def flat_wf_response_ssa():
    """
    Test the response of the SSA WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.

    test the optimised wavefront by checking if it has irregularities.
    Since a flat wavefront at 0 or pi have the same effect, the absolute value of the front is irrelevant.
    """
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
    """
    Test the performance of the Fourier-based algorithm.
    The procedure should significantly increase the signal strength
    """
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera() / 255) * 2 * np.pi
    sim.set_ideal_wf(ideal_wf)
    sim.set_data(ideal_wf)
    sim.trigger()

    controller = Controller(detector=roi_detector, slm=sim)
    alg = BasicFDR(k_angles_min=-1, k_angles_max=1, phase_steps=3, overlap=0.1, controller=controller)
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
    print(f'Enhancement for 9 Fourier modes is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}")
    else:
        return True


import time


def enhancement_ssa():
    """
    Test the performance of the SSA algorithm.
    The procedure should significantly increase the signal strength
    """
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera() / 255) * 2 * np.pi
    sim.set_ideal_wf(ideal_wf)
    # s1 = SLM(0, left=0, width=300, height=300) # input in controller for checking pattern generation

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=3, n_y=3, phase_steps=3, controller=controller)
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
    print(f'Enhancement for 9 SSA modes is {(enhancement / enhancement_perfect) * 100} % of possible enhancement')
    if enhancement < 3:
        raise Exception(
            f"SSA algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}")
    else:
        return True


def square_selection_detector_test():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    width = 20
    height = 20
    detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
    detector.trigger()  #
    if sim.read()[250, 250] / (width * height) != detector.read():
        raise Exception(f"Square detector not working as expected")
    return True


def drawing_detector():
    sim = SimulatedWFS()
    sim.read = data.camera  # overriding the read function for more meaningful images

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


if __name__ == '__main__':

    # Test the performance of the WFS algorithms.
    test_wfs_performance = True

    # Test the different detectors.
    test_detectors = True

    if test_wfs_performance:
        # The following tests test the performance of the WFS algorithms.
        print(flat_wf_response_ssa())
        print(flat_wf_response_fourier())
        print(enhancement_fourier())
        print(enhancement_ssa())

    if test_detectors:
        # The following tests test the different detectors
        print(square_selection_detector_test())
        print(drawing_detector())
