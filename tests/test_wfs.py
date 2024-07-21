import astropy.units as u
import numpy as np
import pytest
import skimage
from scipy.ndimage import zoom
from skimage.transform import resize

from ..openwfs.algorithms import StepwiseSequential, FourierDualReference, FourierDualReferenceCircle, \
    CustomBlindDualReference, troubleshoot
from ..openwfs.algorithms.utilities import WFSController
from ..openwfs.algorithms.troubleshoot import field_correlation
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS, StaticSource, SLM, Microscope, ADCProcessor, Shutter
from ..openwfs.utilities import set_pixel_size, tilt


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
    sim = SimulatedWFS(aberrations=aberrations)
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
    generator = np.random.default_rng(seed=12345)
    aberrations = generator.uniform(0.0, 2 * np.pi, (n_y, n_x))
    sim_no_noise = SimulatedWFS(aberrations=aberrations)
    slm = sim_no_noise.slm
    scale = np.max(sim_no_noise.read())
    sim = ADCProcessor(sim_no_noise, analog_max=scale * 200.0, digital_max=10000, shot_noise=True, generator=generator)
    alg = StepwiseSequential(feedback=sim, slm=slm, n_x=n_x, n_y=n_y, phase_steps=10)
    result = alg.execute()
    print(result.fidelity_noise)

    assert_enhancement(slm, sim, result)


def test_ssa_enhancement():
    input_shape = (40, 40)
    output_shape = (200, 200)  # todo: resize
    rng = np.random.default_rng(seed=12345)

    def get_random_aberrations():
        return resize(rng.uniform(size=input_shape) * 2 * np.pi, output_shape, order=0)

    # Define mock hardware and algorithm
    slm = SLM(shape=output_shape)

    # Find average background intensity
    unshaped_intensities = np.zeros((30,))
    for n in range(len(unshaped_intensities)):
        signal = SimulatedWFS(aberrations=get_random_aberrations(), slm=slm)
        unshaped_intensities[n] = signal.read()

    num_runs = 10
    shaped_intensities_ssa = np.zeros(num_runs)
    for r in range(num_runs):
        sim = SimulatedWFS(aberrations=get_random_aberrations(), slm=slm)

        # SSA
        print(f'SSA run {r + 1}/{num_runs}')
        alg_ssa = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=13, n_y=13, phase_steps=6)
        wfs_result_ssa = alg_ssa.execute()
        sim.slm.set_phases(-np.angle(wfs_result_ssa.t))
        shaped_intensities_ssa[r] = sim.read()

    # Compute enhancements and error margins
    enhancement_ssa = shaped_intensities_ssa.mean() / unshaped_intensities.mean()
    enhancement_ssa_std = shaped_intensities_ssa.std() / unshaped_intensities.mean()

    print(
        f'SSA enhancement (squared signal): {enhancement_ssa:.2f}, std={enhancement_ssa_std:.2f}, with {wfs_result_ssa.n} modes')

    assert enhancement_ssa > 100.0


@pytest.mark.parametrize("n_x", [2, 3])
def test_fourier(n_x):
    """
    Test the enhancement performance of the Fourier-based algorithm.
    Use the 'cameraman' test image since it is relatively smooth.
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-n_x,
                               k_angles_max=n_x,
                               phase_steps=4)
    results = alg.execute()
    assert_enhancement(sim.slm, sim, results, np.exp(1j * aberrations))


def test_fourier2():
    """Test the Fourier dual reference algorithm using WFSController."""
    slm_shape = (1000, 1000)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)
    sim = SimulatedWFS(aberrations=aberrations)
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
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_angles_min=-32,
                               k_angles_max=32,
                               phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    scaled_aberration = zoom(aberrations, np.array(slm_shape) / aberrations.shape)
    assert_enhancement(sim.slm, sim, controller._result, np.exp(1j * scaled_aberration))


@pytest.mark.parametrize("k_radius, g", [[2.5, (1.0, 0.0)], [2.5, (0.0, 2.0)]],)
def test_fourier_circle(k_radius, g):
    """
    Test Fourier dual reference algorithm with a circular k-space, with a tilt 'aberration'.
    """
    aberrations = tilt(shape=(100, 100), extent=(2, 2), g=g, phase_offset=0.5)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReferenceCircle(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_radius=k_radius,
                                     phase_steps=4)
    results = alg.execute()
    assert_enhancement(sim.slm, sim, results, np.exp(1j * aberrations))


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
    sim = SimulatedWFS(aberrations=aberrations)
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
    sim = SimulatedWFS(aberrations=aberrations)
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
    sim = SimulatedWFS(aberrations=aberrations.reshape((*aberrations.shape, 1)))

    from ..openwfs.algorithms.basic_fourier import FourierDualReferenceCircle
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_angles_min=-1,
                               k_angles_max=1, phase_steps=3)

    t = alg.execute().t

    # test the optimized wavefront by checking if it has irregularities.
    assert np.std(t) < 0.001  # The measured wavefront is not flat.


def test_flat_wf_response_ssa():
    """
    Test the response of the SSA WFS method when the solution is flat.
    """
    aberrations = np.zeros(shape=(512, 512))
    sim = SimulatedWFS(aberrations=aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=4, n_y=4, phase_steps=3)

    # Execute the SSA algorithm to get the optimized wavefront
    t = alg.execute().t
    optimised_wf = np.angle(t)

    # Assert that the standard deviation of the optimized wavefront is below the threshold,
    # indicating that it is effectively flat
    assert np.std(optimised_wf) < 0.001, f"Response flat wavefront not flat, std: {np.std(optimised_wf)}"


def test_multidimensional_feedback_ssa():
    aberrations = np.random.uniform(0.0, 2 * np.pi, (256, 256, 5, 2))
    sim = SimulatedWFS(aberrations=aberrations)

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
    sim = SimulatedWFS(aberrations=aberrations)

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


def test_custom_blind_dual_reference():
    do_debug = False

    # Create set of plane wave modes
    N1 = 12
    N2 = 6
    M = N1 * N2
    mode_set_half = np.flip(np.fft.fft2(np.eye(M).reshape((N1, N2, M)), axes=(0, 1)), axis=1)
    mode_set = np.concatenate((mode_set_half, np.zeros(shape=(N1, N2, M))), axis=1)
    mask = np.concatenate((np.zeros((N1, N2)), np.ones((N1, N2))), axis=1)

    if do_debug:
        # Plot the modes
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        for m in range(M):
            plt.subplot(N2, N1, m + 1)
            plt.imshow(np.angle(mode_set[:, :, m]), vmin=-np.pi, vmax=np.pi)
            plt.title(f'm={m}')
            plt.xticks([])
            plt.yticks([])
        plt.pause(0.1)

    # Create aberrations
    x = np.linspace(-1, 1, 1*N1).reshape((1, -1))
    y = np.linspace(-1, 1, 1*N1).reshape((-1, 1))
    aberrations = (np.sin(0.8*np.pi * x) * np.cos(1.3*np.pi*y) * (1.0*np.pi + 0.6*x + 0.6*y)) % (2*np.pi)
    aberrations[0:3, :] = 0
    aberrations[:, 0:3] = 0

    sim = SimulatedWFS(aberrations=aberrations.reshape((*aberrations.shape, 1)))

    alg = CustomBlindDualReference(feedback=sim, slm=sim.slm, slm_shape=aberrations.shape,
        modes=(mode_set, np.flip(mode_set, axis=1)), set1_mask=mask, phase_steps=4, iterations=4)

    result = alg.execute()

    if do_debug:
        plt.figure()
        plt.imshow(np.angle(np.exp(1j*aberrations)), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('Aberrations')

        plt.figure()
        plt.imshow(np.angle(result.t), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('t')
        plt.colorbar()
        plt.show()

    assert field_correlation(np.exp(1j*aberrations), result.t) > 0.999
