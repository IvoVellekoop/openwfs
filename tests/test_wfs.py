import astropy.units as u
import numpy as np
import pytest
import skimage
from scipy.ndimage import zoom

from ..openwfs.algorithms import StepwiseSequential, FourierDualReference, WFSController, troubleshoot
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS, StaticSource, SLM, Microscope, ADCProcessor, Shutter
from ..openwfs.utilities import set_pixel_size


def assert_enhancement(slm, feedback, wfs_results, t_correct=None):
    """Helper function to check if the intensity in the target focus increases as much as expected"""
    optimised_wf = -np.angle(wfs_results.t)
    slm.set_phases(0.0)
    before = feedback.read()
    slm.set_phases(optimised_wf)
    after = feedback.read()
    ratio = after / before
    estimated_ratio = wfs_results.estimated_optimized_intensity / before
    print(f"expected: {estimated_ratio}, actual: {ratio}")
    assert estimated_ratio * 0.5 <= ratio <= estimated_ratio * 2.0, f"""
        The SSA algorithm did not enhance the focus as much as expected.
        Expected at least 0.5 * {estimated_ratio}, got {ratio}"""

    if t_correct is not None:
        # Check if we correctly measured the transmission matrix.
        # The correlation will be less for fewer segments, hence an (ad hoc) factor of 2/sqrt(n)
        t = wfs_results.t[:]
        corr = np.abs(np.vdot(t_correct, t) / np.sqrt(np.vdot(t_correct, t_correct) * np.vdot(t, t)))
        assert corr > 1.0 - 2.0 / np.sqrt(wfs_results.n)


@pytest.mark.parametrize("n_y, n_x", [(5, 5), (7, 11), (6, 4), (30, 20)])
def test_ssa(n_y, n_x):
    """
    Test the enhancement performance of the SSA algorithm.
    Note, for low N, the improvement estimate is not accurate,
    and the test may sometimes fail due to statistical fluctuations.
    """
    aberrations = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    sim = SimulatedWFS(aberrations)
    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=n_x, n_y=n_y, phase_steps=4)
    result = alg.execute()
    print(np.mean(np.abs(result.t)))
    assert_enhancement(sim.slm, sim, result, np.exp(1j * aberrations))


@pytest.mark.parametrize("n_y, n_x", [(5, 5), (7, 11), (6, 4)])
def test_ssa_noise(n_y, n_x):
    """
    Test the enhancement prediction with noisy SSA.

    Note: this test fails if a smooth image is shown, indicating that the estimators
    only work well for strong scattering at the moment.
    """
    #    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    aberrations = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    sim_no_noise = SimulatedWFS(aberrations)
    slm = sim_no_noise.slm
    scale = np.max(sim_no_noise.read())
    sim = ADCProcessor(sim_no_noise, analog_max=scale * 200.0, digital_max=10000, shot_noise=True)
    alg = StepwiseSequential(feedback=sim, slm=slm, n_x=n_x, n_y=n_y, phase_steps=10)
    result = alg.execute()
    print(result.fidelity_noise)

    assert_enhancement(slm, sim, result)


@pytest.mark.parametrize("n_x", [2, 3])
def test_fourier(n_x):
    """
    Test the enhancement performance of the Fourier-based algorithm.
    Use the 'cameraman' test image since it is relatively smooth.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-n_x,
                               k_angles_max=n_x,
                               phase_steps=4)
    results = alg.execute()
    assert_enhancement(sim.slm, sim, results, np.exp(1j * aberrations))


def test_fourier2():
    """Test the Fourier dual reference algorithm using WFSController."""
    slm_shape = (1000, 1000)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_angles_min=-5,
                               k_angles_max=5,
                               phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    scaled_aberration = zoom(aberrations, np.array(slm_shape) / aberrations.shape)
    assert_enhancement(sim.slm, sim, controller._result, np.exp(1j * scaled_aberration))


@pytest.mark.skip(reason="This test is is not passing yet and needs further inspection to see if the test itself is "
                         "correct.")
def test_fourier3():
    """Test the Fourier dual reference algorithm using WFSController."""
    slm_shape = (32, 32)
    aberrations = np.random.uniform(0.0, 2 * np.pi, slm_shape)
    sim = SimulatedWFS(aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_angles_min=-32,
                               k_angles_max=32,
                               phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    scaled_aberration = zoom(aberrations, np.array(slm_shape) / aberrations.shape)
    assert_enhancement(sim.slm, sim, controller._result, np.exp(1j * scaled_aberration))


def test_fourier_microscope():
    aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
    aberration = StaticSource(aberration_phase, pixel_size=2.0 / np.array(aberration_phase.shape))
    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (250, 250)
    img[signal_location] = 100
    slm_shape = (1000, 1000)

    src = StaticSource(img, 400 * u.nm)
    slm = SLM(shape=(1000, 1000))
    sim = Microscope(source=src, incident_field=slm.field, magnification=1, numerical_aperture=1,
                     aberrations=aberration,
                     wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=100)
    roi_detector = SingleRoi(cam, pos=(250, 250))  # Only measure that specific point
    alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=slm_shape, k_angles_min=-1, k_angles_max=1,
                               phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.FLAT_WAVEFRONT
    before = roi_detector.read()
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    after = roi_detector.read()
    # imshow(controller._optimized_wavefront)
    print(after / before)
    scaled_aberration = zoom(aberration_phase, np.array(slm_shape) / aberration_phase.shape)
    assert_enhancement(slm, roi_detector, controller._result, np.exp(1j * scaled_aberration))


def test_fourier_correction_field():
    """
    Check the field correlation between set aberration and optimized wavefront of the Fourier-based algorithm.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-2,
                               k_angles_max=2,
                               phase_steps=3)
    t = alg.execute().t

    t_correct = np.exp(1j * aberrations)
    correlation = np.vdot(t, t_correct) / np.sqrt(np.vdot(t, t) * np.vdot(t_correct, t_correct))

    # TODO: integrate with other test cases, duplication
    assert abs(correlation) > 0.75


def test_phase_shift_correction():
    """
    Test the effect of shifting the found correction of the Fourier-based algorithm.
    Without the bug, a phase shift of the entire correction should not influence the measurement.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1,
                               k_angles_max=1,
                               phase_steps=3)
    t = alg.execute().t

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t)
    sim.slm.set_phases(0)
    before = sim.read()

    optimised_wf -= 5
    signals = []

    for n in range(5):
        optimised_wf += 2
        sim.slm.set_phases(optimised_wf)
        signal = sim.read()
        signals.append(signal)

    assert np.std(signals) < 0.0001 * before, f"""The simulated response of the Fourier algorithm is sensitive to a 
    flat 
        phase-shift. This is incorrect behaviour"""


def test_flat_wf_response_fourier():
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.

    test the optimized wavefront by checking if it has irregularities.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(aberrations.reshape((*aberrations.shape, 1)))

    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1,
                               k_angles_max=1,
                               phase_steps=3)

    t = alg.execute().t

    # test the optimized wavefront by checking if it has irregularities.
    assert np.std(t) < 0.001  # The measured wavefront is not flat.


def test_flat_wf_response_ssa():
    """
    Test the response of the SSA WFS method when the solution is flat.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=4, n_y=4, phase_steps=3)

    # Execute the SSA algorithm to get the optimized wavefront
    t = alg.execute().t
    optimised_wf = np.angle(t)

    # Assert that the standard deviation of the optimized wavefront is below the threshold,
    # indicating that it is effectively flat
    assert np.std(optimised_wf) < 0.001, f"Response flat wavefront not flat, std: {np.std(optimised_wf)}"


def test_multidimensional_feedback_ssa():
    aberrations = np.random.uniform(0.0, 2 * np.pi, (256, 256, 5, 2))
    sim = SimulatedWFS(aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm)
    t = alg.execute().t

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
    alg = FourierDualReference(feedback=sim, slm=sim.slm, k_angles_min=-1, k_angles_max=1, phase_steps=3)
    t = alg.execute().t

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t[:, :, 2, 1])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert enhancement[2, 1] >= 3.0, f"""The algorithm did not enhance the focus as much as expected.
            Expected at least 3.0, got {enhancement}"""

@pytest.mark.parametrize("gaussian_noise_std", (0.0, 0.1, 0.5, 3.0))
def test_ssa_fidelity(gaussian_noise_std):
    """Test fidelity prediction for WFS simulation with various noise levels."""
    # === Define virtual devices for a WFS simulation ===
    # Define aberration as a pattern of random phases at the pupil plane
    aberrations = np.random.uniform(size=(80, 80)) * 2 * np.pi

    # Define specimen as an image with several bright pixels
    specimen_img = np.zeros((240, 240))
    specimen_img[120, 120] = 2e5
    specimen = set_pixel_size(specimen_img, pixel_size=100 * u.nm)

    # The SLM is conjugated to the back pupil plane
    slm = SLM(shape=(80, 80))
    # Also simulate a shutter that can turn off the light
    shutter = Shutter(slm.field)

    # Simulate a WFS microscope looking at the specimen
    sim = Microscope(source=specimen, incident_field=shutter, aberrations=aberrations, wavelength=800 * u.nm)

    # Simulate a camera device with gaussian noise and shot noise
    cam = sim.get_camera(analog_max=1e4, shot_noise=False, gaussian_noise_std=gaussian_noise_std)

    # Define feedback as circular region of interest in the center of the frame
    roi_detector = SingleRoi(cam, radius=1)

    # === Run wavefront shaping experiment ===
    # Use the stepwise sequential (SSA) WFS algorithm
    n_x = 10
    n_y = 10
    alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=n_x, n_y=n_y, phase_steps=8)

    # Define a region of interest to determine average speckle intensity
    roi_background = SingleRoi(cam, radius=50)

    # Run WFS troubleshooter and output a report to the console
    trouble = troubleshoot(algorithm=alg, background_feedback=roi_background,
                           frame_source=cam, shutter=shutter)

    assert np.isclose(trouble.measured_enhancement, trouble.expected_enhancement, rtol=0.2)
