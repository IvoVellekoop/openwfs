import numpy as np
import astropy.units as u
import pytest

from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS, MockSource, MockSLM, Microscope
from ..openwfs.algorithms import StepwiseSequential
from ..openwfs.algorithms.troubleshoot \
    import cnr, signal_std, find_pixel_shift, field_correlation, frame_correlation, \
    analyze_phase_calibration, measure_modulated_light, measure_modulated_light_dual_phase_stepping


def test_signal_std():
    """
    Test signal std, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.rand(400, 400)
    B = np.random.rand(400, 400)
    assert signal_std(A, A) < 1e-6  # Test noise only
    assert np.abs(signal_std(A + B, B) - A.std()) < 0.005  # Test signal+uncorrelated noise
    assert np.abs(signal_std(A + A, A) - np.sqrt(3) * A.std()) < 0.005  # Test signal+correlated noise


def test_cnr():
    """
    Test Contrast to Noise Ratio, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.randn(800, 800)
    B = np.random.randn(800, 800)
    cnr_gt = 3.0  # Ground Truth
    assert cnr(A, A) < 1e-6  # Test noise only
    assert np.abs(cnr(cnr_gt * A + B, B) - cnr_gt) < 0.01  # Test signal+uncorrelated noise
    assert np.abs(cnr(cnr_gt * A + A, A) - np.sqrt((cnr_gt + 1) ** 2 - 1)) < 0.01  # Test signal+correlated noise


def test_find_pixel_shift():
    """
    Test finding pixel shifts between two images.
    """
    # Define pixel shifts to test
    ABshift_gt = (-3, -5)
    CDshift_gt = (2, -4)

    # Define random image 2D arrays
    A = np.random.rand(20, 19)
    B = np.roll(A, shift=ABshift_gt, axis=(0, 1))
    C = np.random.rand(17, 18)
    D = np.roll(C, shift=CDshift_gt, axis=(0, 1))

    # Find pixel shifts
    AAshift_found = find_pixel_shift(A, A)
    ABshift_found = find_pixel_shift(A, B)
    CCshift_found = find_pixel_shift(C, C)
    CDshift_found = find_pixel_shift(C, D)

    # Assert shifts
    assert AAshift_found == (0, 0)
    assert ABshift_found == ABshift_gt
    assert CCshift_found == (0, 0)
    assert CDshift_found == CDshift_gt


def test_field_correlation():
    """
    Test the field correlation, i.e. g_1 normalized first order correlation function.
    """
    a = np.zeros(shape=(2, 3))
    a[1, 2] = 2.0
    a[0, 0] = 0.0

    b = np.ones(shape=(2, 3))
    b[1, 2] = 0.0
    b[0, 0] = 0.0

    c = np.ones(shape=(2, 3), dtype=np.cdouble)
    c[1, 0] = 1 + 1j
    c[0, 1] = 2 - 3j

    assert field_correlation(a, a) == 1.0  # Self-correlation
    assert field_correlation(2 * a, a) == 1.0  # Invariant under scalar-multiplication
    assert field_correlation(a, b) == 0.0  # Orthogonal arrays
    assert np.abs(field_correlation(a + b, b) - np.sqrt(0.5)) < 1e-10  # Self+orthogonal array
    assert np.abs(field_correlation(b, c) - np.conj(field_correlation(c, b))) < 1e-10  # Arguments swapped


def test_frame_correlation():
    """
    Test the frame correlation, i.e. g_2 normalized second order correlation function.
    Test the following:
        g_2 correlation with self == 1/3 for distribution from random.rand
        g_2 correlation with other == 0
    """
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    assert np.abs(frame_correlation(a, a) - 1 / 3) < 2e-3
    assert np.abs(frame_correlation(a, b)) < 2e-3


def phase_response_test_function(phi, b, c, gamma):
    """A synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return np.clip(2 * np.pi * (b + c * (phi / (2 * np.pi)) ** gamma), 0, None)


def inverse_phase_response_test_function(f, b, c, gamma):
    """Inverse of the synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return 2 * np.pi * ((f / (2 * np.pi) - b) / c) ** (1 / gamma)


def lookup_table_test_function(f, b, c, gamma):
    """
    Compute the lookup indices (i.e. a lookup table)
    for countering the synthetic phase response test function: 2π*(b + c*(phi/2π)^gamma).
    """
    phase = inverse_phase_response_test_function(f, b, c, gamma)
    return (np.mod(phase, 2 * np.pi) * 256 / (2 * np.pi) + 0.5).astype(np.uint8)


@pytest.mark.parametrize("n_y, n_x, phase_steps, b, c, gamma",
                         [(11, 9, 8, -0.05, 1.5, 0.8), (4, 4, 10, -0.05, 1.5, 0.8)])
def test_fidelity_phase_calibration_ssa_noise_free(n_y, n_x, phase_steps, b, c, gamma):
    """
    Test computing phase calibration fidelity factor, with the SSA algorithm. Noise-free scenarios.
    """
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    sim = SimulatedWFS(aberrations)
    alg = StepwiseSequential(feedback=sim, slm=sim.slm, n_x=n_x, n_y=n_y, phase_steps=phase_steps)
    result = alg.execute()
    # fidelity_phase_cal_perfect = analyze_phase_calibration(result)
    assert result.calibration_fidelity > 0.99

    # SLM with incorrect phase response, noise-free
    linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)
    sim.slm.phase_response = phase_response_test_function(linear_phase, b, c, gamma)
    result = alg.execute()
    # fidelity_wrong_phase_response = analyze_phase_calibration(result)
    assert result.calibration_fidelity < 0.9

    # SLM calibrated with phase response corrected by LUT, noise-free
    sim.slm.lookup_table = lookup_table_test_function(linear_phase, b, c, gamma)
    result = alg.execute()
    # fidelity_phase_cal_lut = analyze_phase_calibration(result)
    assert result.calibration_fidelity > 0.99


@pytest.mark.parametrize("n_y, n_x, phase_steps, gaussian_noise_std", [(4, 4, 10, 0.2), (6, 6, 12, 1.0)])
def test_fidelity_phase_calibration_ssa_with_noise(n_y, n_x, phase_steps, gaussian_noise_std):
    """
    Test estimation of phase calibration fidelity factor, with the SSA algorithm. With noise.
    """
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    numerical_aperture = 1.0
    aberration_phase = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 250
    src = MockSource(img, 500 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = MockSLM(shape=(80, 80))
    sim = Microscope(source=src, incident_field=slm.get_monitor('field'), magnification=1,
                     numerical_aperture=numerical_aperture,
                     aberrations=aberration, wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=1e4, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Define and run WFS algorithm
    alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=n_x, n_y=n_y, phase_steps=phase_steps)
    result_good = alg.execute()
    print(result_good.calibration_fidelity)
    fidelity_phase_cal_noise = analyze_phase_calibration(result_good)
    assert fidelity_phase_cal_noise > 0.9

    # SLM with incorrect phase response
    linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)
    slm.phase_response = phase_response_test_function(linear_phase, b=0.05, c=0.6, gamma=1.5)
    result_good = alg.execute()
    print(result_good.calibration_fidelity)
    fidelity_phase_cal_noise = analyze_phase_calibration(result_good)
    assert fidelity_phase_cal_noise < 0.9


@pytest.mark.parametrize("num_blocks, phase_steps, expected_fid, atol", [(10, 8, 1, 1e-6)])
def test_measure_modulated_light_dual_phase_stepping_noise_free(num_blocks, phase_steps, expected_fid, atol):
    """Test fidelity estimation due to amount of modulated light. Noise-free."""
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (20, 20))
    sim = SimulatedWFS(aberrations)

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light_dual_phase_stepping(
        slm=sim.slm, feedback=sim, phase_steps=phase_steps, num_blocks=num_blocks)
    assert np.isclose(fidelity_modulated, expected_fid, atol=atol)


@pytest.mark.parametrize("num_blocks, phase_steps, gaussian_noise_std, atol", [(10, 6, 0.0, 1e-6), (6, 8, 2.0, 1e-3)])
def test_measure_modulated_light_dual_phase_stepping_with_noise(num_blocks, phase_steps, gaussian_noise_std, atol):
    """Test fidelity estimation due to amount of modulated light. Can test with noise."""
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 100
    src = MockSource(img, 200 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = MockSLM(shape=(100, 100))
    sim = Microscope(source=src, incident_field=slm.get_monitor('field'), magnification=1, numerical_aperture=1.0,
                     wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=1e4, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light_dual_phase_stepping(
        slm=slm, feedback=roi_detector, phase_steps=phase_steps, num_blocks=num_blocks)
    assert np.isclose(fidelity_modulated, 1, atol=atol)


@pytest.mark.parametrize("phase_steps, expected_fid, atol, modulated_field_amplitude, non_modulated_field",
                         [(6, 1, 1e-6, 1.0, 0.0), (8, 0.5, 1e-6, 0.5, 0.5), (8, 1 / (1 + 0.25), 1e-6, 1.0, 0.5)])
def test_measure_modulated_light_noise_free(
        phase_steps, expected_fid, atol, modulated_field_amplitude, non_modulated_field):
    """Test fidelity estimation due to amount of modulated light. Noise-free."""
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (20, 20))
    sim = SimulatedWFS(aberrations)
    sim.slm.field.read().modulated_field_amplitude = modulated_field_amplitude
    sim.slm.field.read().non_modulated_field = non_modulated_field

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light(slm=sim.slm, feedback=sim, phase_steps=phase_steps)
    assert np.isclose(fidelity_modulated, expected_fid, atol=atol)


@pytest.mark.parametrize(
    "phase_steps, gaussian_noise_std, expected_fid, atol, modulated_field_amplitude, non_modulated_field",
    [(6, 0.0, 1, 1e-6, 1.0, 0.0), (8, 0.0, 0.5, 1e-6, 0.5, 0.5), (12, 2.0, 1 / (1 + 0.25), 1e-3, 1.0, 0.5)])
def test_measure_modulated_light_dual_phase_stepping_with_noise(
        phase_steps, gaussian_noise_std, expected_fid, atol, modulated_field_amplitude, non_modulated_field):
    """Test fidelity estimation due to amount of modulated light. Can test with noise."""
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 100
    src = MockSource(img, 200 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = MockSLM(shape=(100, 100),
                  field_amplitude=modulated_field_amplitude,
                  non_modulated_field_fraction=non_modulated_field)
    sim = Microscope(source=src, incident_field=slm.get_monitor('field'), magnification=1, numerical_aperture=1.0,
                     wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=1e4, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light(slm=slm, feedback=roi_detector, phase_steps=phase_steps)
    assert np.isclose(fidelity_modulated, expected_fid, atol=atol)
