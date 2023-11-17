import matplotlib.pyplot as plt

from ..openwfs.simulation import SimulatedWFS, MockSource, MockSLM, Microscope
import numpy as np
from ..openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from ..openwfs.processors import SingleRoi
import skimage
import astropy.units as u
import cv2


def calculate_enhancement(simulation, optimised_wf, x=256, y=256):
    simulation.set_data(0)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_before = np.mean(simulation.read()[x, y])

    simulation.set_data(optimised_wf)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_after = np.mean(simulation.read()[x, y])

    return feedback_after / feedback_before


def test_ssa():
    """
    Test the enhancement performance of the SSA algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)
    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = StepwiseSequential(feedback=roi_detector, slm=sim.slm, n_x=3, n_y=3, phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t[..., 0])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = roi_detector.read()
    sim.slm.set_phases(optimised_wf)
    after = roi_detector.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The SSA algorithm did not enhance focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)
    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = BasicFDR(feedback=roi_detector, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = np.angle(t)

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = roi_detector.read()
    sim.slm.set_phases(optimised_wf)
    after = roi_detector.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The Fourier algorithm did not enhance focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_fourier_correction_field():
    """
    Check the field correlation between set aberration and optimised wavefront of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)
    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = BasicFDR(feedback=roi_detector, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)
    t = alg.execute()

    norm_t1 = np.exp(1j * -aberrations) / (len(aberrations[:]))
    norm_t2 = t / np.linalg.norm(t[:])
    # If you need to show the phase difference between the two fields:
    # plt.imshow(np.angle(norm_t1/norm_t2))
    # plt.show()

    assert abs(
        np.vdot(norm_t1, norm_t2)) > 0.73  # not that high because a 9 mode WFS procedure can only do so much


def test_pathfinding_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)
    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = CharacterisingFDR(feedback=roi_detector, slm=sim.slm, phase_steps=3, overlap=0.1, max_modes=12)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = np.angle(t)

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = roi_detector.read()
    sim.slm.set_phases(optimised_wf)
    after = roi_detector.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The SSA algorithm did not enhance focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_phaseshift_correction():
    """
    Test the effect of shifting the found correction of the Fourier-based algorithm.
    Arose from bug, a phaseshift of the entire correction should not influence the measurement.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)
    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = BasicFDR(feedback=roi_detector, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = np.angle(t)
    sim.slm.set_phases(0)
    before = roi_detector.read()

    optimised_wf -= 5
    signals = []

    for n in range(5):
        optimised_wf += 2
        sim.slm.set_phases(optimised_wf)
        signal = roi_detector.read()
        signals.append(signal)

    enhancements = signals / before

    assert np.std(enhancements) < 0.0001, f"""The simulated response of the Fourier algorithm is sensitive to a flat 
        phase-shift. This is incorrect behaviour"""


def test_flat_wf_response_fourier():
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.

    test the optimised wavefront by checking if it has irregularities.
    Since a flat wavefront at 0 or pi have the same effect, the absolute value of the front is irrelevant.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)

    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = BasicFDR(feedback=roi_detector, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)

    t = alg.execute()
    optimised_wf = np.angle(t)

    # test the optimised wavefront by checking if it has irregularities. Since a flat wavefront at 0 or pi
    assert np.std(optimised_wf) < 0.001  # "Response flat wavefront not flat"


def test_flat_wf_response_ssa():
    """
    Test the response of the SSA WFS method when the solution is flat.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)

    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = StepwiseSequential(feedback=roi_detector, slm=sim.slm, n_x=4, n_y=4, phase_steps=3)

    # Execute the SSA algorithm to get the optimized wavefront
    t = alg.execute()
    optimised_wf = np.angle(t)

    # Assert that the standard deviation of the optimized wavefront is below the threshold,
    # indicating that it is effectively flat
    assert np.std(optimised_wf) < 0.001, f"Response flat wavefront not flat, std: {np.std(optimised_wf)}"


def test_flat_wf_response_pathfinding_fourier():
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.

    test the optimised wavefront by checking if it has irregularities.
    Since a flat wavefront at 0 or pi have the same effect, the absolute value of the front is irrelevant.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(width=512, height=512, aberrations=aberrations)

    roi_detector = SingleRoi(sim.cam, x=256, y=256, radius=0.5)
    alg = CharacterisingFDR(feedback=roi_detector, slm=sim.slm, phase_steps=3, overlap=0.1, max_modes=12)

    # Execute the SSA algorithm to get the optimized wavefront
    t = alg.execute()
    optimised_wf = np.angle(t)

    # Assert that the standard deviation of the optimized wavefront is below the threshold,
    # indicating that it is effectively flat
    assert np.std(optimised_wf) < 0.001, f"Response flat wavefront not flat, std: {np.std(optimised_wf)}"
