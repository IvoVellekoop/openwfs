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


def test_flat_wf_response_ssa():
    """
    Test the response of the SSA WFS method when the solution is flat.
    """
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
    roi_detector = SingleRoi(sim, x=250, y=250, radius=3.0)

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=6, n_y=6, phase_steps=3, controller=controller)

    # Execute algorithm and get the optimised wavefront
    t = alg.execute()
    optimised_wf = np.angle(t)

    # Use pytest's assert to validate the conditions
    assert np.std(optimised_wf) < 0.001, "Response flat wavefront not flat"


def test_flat_wf_response_pathfinding_fourier():
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

    alg = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=12, controller=controller)

    t = alg.execute()
    optimised_wf = np.angle(t)

    # test the optimised wavefront by checking if it has irregularities. Since a flat wavefront at 0 or pi
    assert np.std(optimised_wf) < 0.001  # "Response flat wavefront not flat"


def test_enhancement_pathfinding_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
    """
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera() / 255) * 2 * np.pi
    sim.set_ideal_wf(ideal_wf)
    sim.set_data(ideal_wf)
    sim.trigger()
    controller = Controller(detector=roi_detector, slm=sim)
    alg = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=12, controller=controller)
    t = alg.execute()
    optimised_wf = np.angle(t)

    # Calculation of the enhancement factor
    enhancement = calculate_enhancement(sim, optimised_wf, x=256, y=256)

    # Assert condition for the enhancement factor
    assert enhancement >= 3, f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}"


def test_enhancement_fourier():
    """
    Test the enhancement performance of the Fourier-based algorithm.
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

    # Calculation of the enhancement factor
    enhancement = calculate_enhancement(sim, optimised_wf, x=256, y=256)

    # Assert condition for the enhancement factor
    assert enhancement >= 3, f"Fourier algorithm does not enhance focus as much as expected. Expected 3, got {enhancement}"


def test_enhancement_ssa():
    """
    Test the enhancement performance of the SSA algorithm.
    """
    sim = SimulatedWFS(width=512, height=512)
    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    ideal_wf = (data.camera() / 255) * 2 * np.pi
    sim.set_ideal_wf(ideal_wf)

    controller = Controller(detector=roi_detector, slm=sim)
    alg = StepwiseSequential(n_x=3, n_y=3, phase_steps=3, controller=controller)
    t = alg.execute()

    optimised_wf = np.angle(t)

    # Calculation of the enhancement factor
    enhancement = calculate_enhancement(sim, optimised_wf)
    enhancement_perfect = calculate_enhancement(sim, ideal_wf)

    # Assert condition for the enhancement factor
    assert enhancement >= 3, f"SSA algorithm does not enhance focus as much as expected. Expected at least 3, got {enhancement}"
