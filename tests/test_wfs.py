import astropy.units as u
import numpy as np
import pytest
import skimage
from scipy.linalg import hadamard
from scipy.ndimage import zoom

from openwfs.simulation.mockdevices import GaussianNoise
from ..openwfs.algorithms import StepwiseSequential, FourierDualReference, \
    DualReference, troubleshoot
from ..openwfs.algorithms.troubleshoot import field_correlation
from ..openwfs.algorithms.utilities import WFSController
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS, StaticSource, SLM, Microscope, Shutter
from ..openwfs.utilities import set_pixel_size, tilt


def assert_enhancement(slm, feedback, wfs_results):
    """Helper function to check if the intensity in the target focus increases as much as expected"""
    optimised_wf = -np.angle(wfs_results.t)
    slm.set_phases(0.0)
    before = feedback.read()
    slm.set_phases(optimised_wf)
    after = feedback.read()
    ratio = after / before
    estimated_ratio = wfs_results.estimated_enhancement  # wfs_results.estimated_optimized_intensity / before
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


@pytest.mark.parametrize("shape", [(4, 7), (10, 7), (20, 31)])
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_ssa(shape, noise: float):
    """
    Test the SSA algorithm.

    This tests checks if the algorithm achieves the theoretical enhancement,
    and it also verifies that the enhancement and noise fidelity
    are estimated correctly by the algorithm.
    """
    np.random.seed(42)  # for reproducibility

    M = 100  # number of targets
    phase_steps = 6

    N = np.prod(shape)  # number of input modes
    sim = SimulatedWFS(t=random_transmission_matrix((*shape, M)))
    I_0 = np.mean(sim.read())

    # create feedback object, with noise if needed
    if noise > 0.0:
        sim.slm.set_phases(0.0)
        feedback = GaussianNoise(sim, std=I_0 * noise)
        signal = (N - 1) / N ** 2
        theoretical_noise_fidelity = signal / (signal + noise ** 2 / phase_steps)
    else:
        feedback = sim
        theoretical_noise_fidelity = 1.0

    # Execute the SSA algorithm to get the optimized wavefront
    # for all targets simultaneously
    alg = StepwiseSequential(feedback=feedback, slm=sim.slm, n_x=shape[1], n_y=shape[0], phase_steps=phase_steps)
    alg_fidelity = (N - 1) / N  # SSA is inaccurate if N is low
    result = alg.execute()

    # Determine the optimized intensities in each of the targets individually
    # Also estimate the fidelity of the transmission matrix reconstruction
    # This fidelity is determined row by row, since we need to compensate
    # the unknown phases. The normalization of the correlation function
    # is performed on all rows together, not per row, to increase
    # the accuracy of the estimate.
    I_opt = np.zeros((M,))
    t_correlation = 0.0
    t_norm = 0.0
    for b in range(M):
        sim.slm.set_phases(-np.angle(result.t[:, :, b]))
        I_opt[b] = feedback.read()[b]
        t_correlation += abs(np.vdot(result.t[:, :, b], sim.t[:, :, b])) ** 2
        t_norm += np.vdot(result.t[:, :, b], result.t[:, :, b]) * np.vdot(sim.t[:, :, b], sim.t[:, :, b])
    t_correlation /= t_norm

    # Check the enhancement, noise fidelity and
    # the fidelity of the transmission matrix reconstruction
    enhancement = I_opt.mean() / I_0
    theoretical_enhancement = np.pi / 4 * theoretical_noise_fidelity * alg_fidelity * (N - 1) + 1
    estimated_enhancement = result.estimated_enhancement.mean() * alg_fidelity
    theoretical_t_correlation = theoretical_noise_fidelity * alg_fidelity
    estimated_t_correlation = result.fidelity_noise * result.fidelity_calibration * alg_fidelity
    tolerance = 2.0 / np.sqrt(M)
    print(
        f"\nenhancement:      \ttheoretical= {theoretical_enhancement},\testimated={estimated_enhancement},\tactual: {enhancement}")
    print(
        f"t-matrix fidelity:\ttheoretical = {theoretical_t_correlation},\testimated = {estimated_t_correlation},\tactual = {t_correlation}")
    print(f"noise fidelity:   \ttheoretical = {theoretical_noise_fidelity},\testimated = {result.fidelity_noise}")
    print(f"comparing at relative tolerance: {tolerance}")

    assert np.allclose(enhancement, theoretical_enhancement, rtol=tolerance), f"""
        The SSA algorithm did not enhance the focus as much as expected.
        Theoretical {theoretical_enhancement}, got {enhancement}"""

    assert np.allclose(estimated_enhancement, enhancement, rtol=tolerance), f"""
         The SSA algorithm did not estimate the enhancement correctly.
         Estimated {estimated_enhancement}, got {enhancement}"""

    assert np.allclose(t_correlation, theoretical_t_correlation, rtol=tolerance), f"""
        The SSA algorithm did not measure the transmission matrix correctly.
        Expected {theoretical_t_correlation}, got {t_correlation}"""

    assert np.allclose(estimated_t_correlation, theoretical_t_correlation, rtol=tolerance), f"""
        The SSA algorithm did not estimate the fidelity of the transmission matrix correctly.
        Expected {theoretical_t_correlation}, got {estimated_t_correlation}"""

    assert np.allclose(result.fidelity_noise, theoretical_noise_fidelity, rtol=tolerance), f"""
        The SSA algorithm did not estimate the noise correctly.
        Expected {theoretical_noise_fidelity}, got {result.fidelity_noise}"""


def random_transmission_matrix(shape):
    """
    Create a random transmission matrix with the given shape.
    """
    return np.random.normal(size=shape) + 1j * np.random.normal(size=shape)


@pytest.mark.parametrize("k_radius", [2, 3])
def test_fourier(k_radius):
    """
    Test the enhancement performance of the Fourier-based algorithm.
    Check if the estimated enhancement is close to the actual enhancement.
    Check if the measured transmission matrix is close to the actual transmission matrix.
    For this check, compare two situations: one with a completely random aberration pattern,
    and one with a smooth aberration pattern. In the latter case, the measured transmission matrix
    should match the actual transmission matrix better than for the completely random one
    """
    shape = (16, 15)
    sim = SimulatedWFS(t=random_transmission_matrix(shape))
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=shape, k_radius=k_radius)
    results = alg.execute()
    assert_enhancement(sim, results)


def test_fourier2():
    """Test the Fourier dual reference algorithm using WFSController."""
    slm_shape = (1000, 1000)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_radius=7.5, phase_steps=3)
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
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_radius=45,
                               phase_steps=3)
    controller = WFSController(alg)
    controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
    scaled_aberration = zoom(aberrations, np.array(slm_shape) / aberrations.shape)
    assert_enhancement(sim.slm, sim, controller._result, np.exp(1j * scaled_aberration))


@pytest.mark.parametrize("k_radius, g", [[2.5, (1.0, 0.0)], [2.5, (0.0, 2.0)]], )
def test_fourier_circle(k_radius, g):
    """
    Test Fourier dual reference algorithm with a circular k-space, with a tilt 'aberration'.
    """
    aberrations = tilt(shape=(100, 100), extent=(2, 2), g=g, phase_offset=0.5)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_radius=k_radius,
                               phase_steps=4)

    do_debug = False
    if do_debug:
        # Plot the modes
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        patterns = alg.phase_patterns[0] * np.expand_dims(alg.masks[0], axis=-1)
        N = patterns.shape[2]
        Nsqrt = int(np.ceil(np.sqrt(N)))
        for m in range(N):
            plt.subplot(Nsqrt, Nsqrt, m + 1)
            plt.imshow(np.cos(patterns[:, :, m]), vmin=-1.0, vmax=1.0)
            plt.title(f'm={m}')
            plt.xticks([])
            plt.yticks([])
        plt.colorbar()
        plt.pause(0.01)
        plt.suptitle('Phase of basis functions for one half')

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
    alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=slm_shape, k_radius=1.5,
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
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_radius=3.0,
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
    TODO: move to test of SimulatedWFS, since it is not testing the WFS algorithm itself
    """
    aberrations = skimage.data.camera() * (2.0 * np.pi / 255.0)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_radius=1.5,
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

    assert np.std(signals) / np.mean(signals) < 0.001, f"""The response of SimulatedWFS is sensitive to a flat 
        phase shift. This is incorrect behaviour"""


@pytest.mark.parametrize("optimized_reference", [True, False])
@pytest.mark.parametrize("step", [True, False])
def test_flat_wf_response_fourier(optimized_reference, step):
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.
    Also tests if stitching is done correctly by having an aberration pattern which is flat (but different) on the two halves.

    test the optimized wavefront by checking if it has irregularities.
    """
    aberrations = np.ones(shape=(4, 4))
    if step:
        aberrations[:, 2:] = 2.0
    sim = SimulatedWFS(aberrations=aberrations.reshape((*aberrations.shape, 1)))

    alg = FourierDualReference(feedback=sim, slm=sim.slm, slm_shape=np.shape(aberrations), k_radius=1.5, phase_steps=3,
                               optimized_reference=optimized_reference)

    t = alg.execute().t

    # test the optimized wavefront by checking if it has irregularities.
    measured_aberrations = np.squeeze(np.angle(t))
    measured_aberrations += aberrations[0, 0] - measured_aberrations[0, 0]
    assert np.allclose(measured_aberrations, aberrations, atol=0.02)  # The measured wavefront is not flat.


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
    alg = FourierDualReference(feedback=sim, slm=sim.slm, k_radius=3.5, phase_steps=3)
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


@pytest.mark.parametrize("type", ('plane_wave', 'hadamard'))
@pytest.mark.parametrize("shape", ((8, 8), (6, 4)))
def test_custom_blind_dual_reference_ortho_split(type: str, shape):
    """Test custom blind dual reference with an orthonormal phase-only basis.
    Two types of bases are tested: plane waves and Hadamard"""
    do_debug = False
    N = shape[0] * (shape[1] // 2)
    modes_shape = (shape[0], shape[1] // 2, N)
    if type == 'plane_wave':
        # Create a full plane wave basis for one half of the SLM.
        modes = np.fft.fft2(np.eye(N).reshape(modes_shape), axes=(0, 1))
    else:  # type == 'hadamard':
        modes = hadamard(N).reshape(modes_shape)

    mask = np.concatenate((np.zeros(modes_shape[0:1], dtype=bool), np.ones(modes_shape[0:1], dtype=bool)), axis=1)
    mode_set = np.concatenate((modes, np.zeros(shape=modes_shape)), axis=1)
    phases_set = np.angle(mode_set)

    if do_debug:
        # Plot the modes
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        for m in range(N):
            plt.subplot(*modes_shape[0:1], m + 1)
            plt.imshow(np.angle(mode_set[:, :, m]), vmin=-np.pi, vmax=np.pi)
            plt.title(f'm={m}')
            plt.xticks([])
            plt.yticks([])
        plt.pause(0.1)

    # Create aberrations
    sim = SimulatedWFS(t=random_transmission_matrix(shape))

    alg = DualReference(feedback=sim, slm=sim.slm,
                        phase_patterns=(phases_set, np.flip(phases_set, axis=1)), group_mask=mask,
                        iterations=4)

    result = alg.execute()

    if do_debug:
        plt.figure()
        plt.imshow(np.angle(sim.t), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('Aberrations')

        plt.figure()
        plt.imshow(np.angle(result.t), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('t')
        plt.colorbar()
        plt.show()

    assert np.abs(field_correlation(np.exp(1j * aberrations), result.t)) > 0.999


def test_custom_blind_dual_reference_non_ortho():
    """
    Test custom blind dual reference with a non-orthogonal basis.
    """
    do_debug = False

    # Create set of modes that are barely linearly independent
    N1 = 6
    N2 = 3
    M = N1 * N2
    mode_set_half = (1 / M) * (1j * np.eye(M).reshape((N1, N2, M)) * -np.ones(shape=(N1, N2, M)))
    mode_set = np.concatenate((mode_set_half, np.zeros(shape=(N1, N2, M))), axis=1)
    phases_set = np.angle(mode_set)
    mask = np.concatenate((np.zeros((N1, N2)), np.ones((N1, N2))), axis=1)

    if do_debug:
        # Plot the modes
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        for m in range(M):
            plt.subplot(N2, N1, m + 1)
            plt.imshow(phases_set[:, :, m], vmin=-np.pi, vmax=np.pi)
            plt.title(f'm={m}')
            plt.xticks([])
            plt.yticks([])
        plt.pause(0.01)
        plt.suptitle('Phase of basis functions for one half')

    # Create aberrations
    x = np.linspace(-1, 1, 1 * N1).reshape((1, -1))
    y = np.linspace(-1, 1, 1 * N1).reshape((-1, 1))
    aberrations = (np.sin(0.8 * np.pi * x) * np.cos(1.3 * np.pi * y) * (0.8 * np.pi + 0.4 * x + 0.4 * y)) % (2 * np.pi)
    aberrations[0:1, :] = 0
    aberrations[:, 0:2] = 0

    sim = SimulatedWFS(aberrations=aberrations)

    alg = DualReference(feedback=sim, slm=sim.slm,
                        phase_patterns=(phases_set, np.flip(phases_set, axis=1)), group_mask=mask,
                        phase_steps=4,
                        iterations=4)

    result = alg.execute()

    if do_debug:
        plt.figure()
        plt.imshow(np.angle(np.exp(1j * aberrations)), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('Aberrations')
        plt.colorbar()

        plt.figure()
        plt.imshow(np.angle(result.t), vmin=-np.pi, vmax=np.pi, cmap='hsv')
        plt.title('t')
        plt.colorbar()
        plt.show()

    assert np.abs(field_correlation(np.exp(1j * aberrations), result.t)) > 0.999
