import matplotlib.pyplot as plt

from ..openwfs.simulation import SimulatedWFS, MockSource, MockSLM, Microscope
import numpy as np
from ..openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR, WFSController
from ..openwfs.processors import SingleRoi, CropProcessor
import skimage
from ..openwfs.utilities import imshow
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
    sim = SimulatedWFS(aberrations)
    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=10, n_y=10, phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t)

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The SSA algorithm did not enhance the focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations.reshape(*aberrations.shape, 1))
    alg = BasicFDR(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t[:, :, 0])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The Fourier algorithm did not enhance focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_fourier2():
    """Test to reproduce a bug observed in microscope_simulation_micromanager.py
    and reduced to simplest code to reproduce it."""

    aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0)  # + np.pi
    roi_detector = SimulatedWFS(aberration_phase)
    alg = BasicFDR(feedback=roi_detector, slm=roi_detector.slm, slm_shape=(1000, 1000), k_angles_min=-3, k_angles_max=3,
                   phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.FLAT_WAVEFRONT
    before = roi_detector.read()
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    after = roi_detector.read()
    # imshow(controller._optimized_wavefront)
    print(after / before)
    assert after > 3.0 * before


def test_fourier_microscope():
    aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
    aberration = MockSource(aberration_phase, pixel_size=2.0 / np.array(aberration_phase.shape))
    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (256, 256)
    img[signal_location] = 100
    src = MockSource(img, 400 * u.nm)
    slm = MockSLM(shape=(1000, 1000))
    sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=1, aberrations=aberration,
                     wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=100)
    roi_detector = SingleRoi(cam, x=256, y=256, radius=0)  # Only measure that specific point
    alg = BasicFDR(feedback=roi_detector, slm=slm, slm_shape=(1000, 1000), k_angles_min=-2, k_angles_max=2,
                   phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.FLAT_WAVEFRONT
    before = roi_detector.read()
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    after = roi_detector.read()
    # imshow(controller._optimized_wavefront)
    print(after / before)
    assert after > 3.0 * before


def test_fourier_correction_field():
    """
    Check the field correlation between set aberration and optimised wavefront of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = BasicFDR(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-2, k_angles_max=2,
                   phase_steps=3)
    t = alg.execute()

    t_correct = np.exp(1j * aberrations)
    correlation = np.vdot(t, t_correct) / np.sqrt(np.vdot(t, t) * np.vdot(t_correct, t_correct))

    assert abs(correlation) > 0.75  # not that high because a 9 mode WFS procedure can only do so much


def test_pathfinding_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = CharacterisingFDR(feedback=sim, slm=sim.slm, phase_steps=3, overlap=0.1, max_modes=12)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t)

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement >= 3.0, f"""The SSA algorithm did not enhance focus as much as expected.
        Expected at least 3.0, got {enhancement}"""


def test_phaseshift_correction():
    """
    Test the effect of shifting the found correction of the Fourier-based algorithm.
    Without the bug, a phase shift of the entire correction should not influence the measurement.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = BasicFDR(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
                   phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = np.angle(t)
    sim.slm.set_phases(0)
    before = sim.read()

    optimised_wf -= 5
    signals = []

    for n in range(5):
        optimised_wf += 2
        sim.slm.set_phases(optimised_wf)
        signal = sim.read()
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
    sim = SimulatedWFS(aberrations.reshape((*aberrations.shape, 1)))

    alg = BasicFDR(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1, k_angles_max=1,
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
    sim = SimulatedWFS(aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=4, n_y=4, phase_steps=3)

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
    sim = SimulatedWFS(aberrations.reshape((*aberrations.shape, 1)))
    alg = CharacterisingFDR(feedback=sim, slm=sim.slm, phase_steps=3, overlap=0.1, max_modes=12)

    # Execute the SSA algorithm to get the optimized wavefront
    t = alg.execute()
    optimised_wf = np.angle(t)

    # Assert that the standard deviation of the optimized wavefront is below the threshold,
    # indicating that it is effectively flat
    assert np.std(optimised_wf) < 0.001, f"Response flat wavefront not flat, std: {np.std(optimised_wf)}"


def test_multidimensional_feedback_ssa():
    aberrations = np.random.uniform(0.0, 2 * np.pi, (256, 256, 5, 2))
    sim = SimulatedWFS(aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 2,1
    target = (2, 1)
    optimised_wf = -np.angle(t[(..., *target)])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement[target] >= 3.0, f"""The SSA algorithm did not enhance focus as much as expected.
            Expected at least 3.0, got {enhancement}"""


def test_multidimensional_feedback_fourier():
    aberrations = np.random.uniform(0.0, 2 * np.pi, (256, 256, 5, 2))
    sim = SimulatedWFS(aberrations)

    # input the camera as a feedback object, such that it is multidimensional
    alg = BasicFDR(feedback=sim, slm=sim.slm, k_angles_min=-1, k_angles_max=1, phase_steps=3)
    t = alg.execute()

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = np.angle(t[:, :, 2, 1])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement[2, 1] >= 3.0, f"""The algorithm did not enhance the focus as much as expected.
            Expected at least 3.0, got {enhancement}"""
