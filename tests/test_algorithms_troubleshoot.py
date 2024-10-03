import astropy.units as u
import numpy as np
import pytest

from .test_simulation import phase_response_test_function, lookup_table_test_function
from ..openwfs.algorithms import StepwiseSequential
from ..openwfs.algorithms.troubleshoot import (
    cnr,
    signal_std,
    find_pixel_shift,
    field_correlation,
    frame_correlation,
    pearson_correlation,
    measure_modulated_light,
    measure_modulated_light_dual_phase_stepping,
)
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS, StaticSource, SLM, Microscope


def test_signal_std():
    """
    Test signal std, corrected for (uncorrelated) noise in the signal.
    """
    a = np.random.rand(400, 400)
    b = np.random.rand(400, 400)
    assert signal_std(a, a) < 1e-6  # Test noise only
    assert (
        np.abs(signal_std(a + b, b) - a.std()) < 0.005
    )  # Test signal+uncorrelated noise
    assert (
        np.abs(signal_std(a + a, a) - np.sqrt(3) * a.std()) < 0.005
    )  # Test signal+correlated noise


def test_cnr():
    """
    Test Contrast to Noise Ratio, corrected for (uncorrelated) noise in the signal.
    """
    a = np.random.randn(800, 800)
    b = np.random.randn(800, 800)
    cnr_gt = 3.0  # Ground Truth
    assert cnr(a, a) < 1e-6  # Test noise only
    assert (
        np.abs(cnr(cnr_gt * a + b, b) - cnr_gt) < 0.01
    )  # Test signal+uncorrelated noise
    assert (
        np.abs(cnr(cnr_gt * a + a, a) - np.sqrt((cnr_gt + 1) ** 2 - 1)) < 0.01
    )  # Test signal+correlated noise


def test_find_pixel_shift():
    """
    Test finding pixel shifts between two images.
    """
    # Define pixel shifts to test
    ab_shift = (-3, -5)
    cd_shift = (2, -4)

    # Define random image 2D arrays
    a = np.random.rand(20, 19)
    b = np.roll(a, shift=ab_shift, axis=(0, 1))
    c = np.random.rand(17, 18)
    d = np.roll(c, shift=cd_shift, axis=(0, 1))

    # Find pixel shifts
    aa_shift_found = find_pixel_shift(a, a)
    ab_shift_found = find_pixel_shift(a, b)
    cc_shift_found = find_pixel_shift(c, c)
    cd_shift_found = find_pixel_shift(c, d)

    # Assert shifts
    assert aa_shift_found == (0, 0)
    assert ab_shift_found == ab_shift
    assert cc_shift_found == (0, 0)
    assert cd_shift_found == cd_shift


def test_field_correlation():
    """
    Test the field correlation, i. e. g_1 normalized first order correlation function.
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
    assert (
        np.abs(field_correlation(a + b, b) - np.sqrt(0.5)) < 1e-10
    )  # Self+orthogonal array
    assert (
        np.abs(field_correlation(b, c) - np.conj(field_correlation(c, b))) < 1e-10
    )  # Arguments swapped


def test_frame_correlation():
    """
    Test the frame correlation, i. e. g_2 normalized second order correlation function.
    Test the following:
        g_2 correlation with self == 1/3 for distribution from `random.rand`
        g_2 correlation with other == 0
    """
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    assert np.abs(frame_correlation(a, a) - 1 / 3) < 2e-3
    assert np.abs(frame_correlation(a, b)) < 2e-3


def test_pearson_correlation():
    """
    Test the Pearson correlation.
    """
    # Perfect correlation
    a = np.asarray((1, 2, 3))
    b = np.asarray((2, 4, 6))
    corr_ab = pearson_correlation(a, b)
    corr_minus_ab = pearson_correlation(-a, b)
    assert np.isclose(corr_ab, 1, atol=1e-6)
    assert np.isclose(corr_minus_ab, -1, atol=1e-6)

    # No correlation
    c = np.asarray((1, 0, -1))
    d = np.asarray((0, 2, 0))
    corr_cd_compute = pearson_correlation(c, d)
    assert np.isclose(corr_cd_compute, 0, atol=1e-6)

    # Some correlation
    e = np.asarray((4, -4, 2, -2))
    f = np.asarray((2, -2, -1, 1))
    corr_ef = pearson_correlation(e, f)
    assert np.isclose(corr_ef, 0.6, atol=1e-6)


def test_pearson_correlation_noise_compensated():
    """
    Test Pearson correlation, compensated for noise.

    For Spearman's attenuation factor, see also:
     - Saccenti et al. 2020, https://www.nature.com/articles/s41598-019-57247-4
     - https://en.wikipedia.org/wiki/Regression_dilution#Correlation_correction
    """
    N = 1000000
    a = np.random.rand(N)
    b = np.random.rand(N)
    noise1 = np.random.rand(N)
    noise2 = np.random.rand(N)

    # Generate fake signals
    A1 = 3 * a
    A2 = 4 * a
    B2 = 5 * b
    A_noisy1 = A1 + noise1
    A_noisy2 = A2 + noise2
    B_noisy2 = B2 + noise2

    corr_AA = pearson_correlation(A_noisy1, A_noisy2, noise_var=noise1.var())
    corr_AB = pearson_correlation(A_noisy1, B_noisy2, noise_var=noise1.var())
    corr_AA_with_noise = pearson_correlation(A_noisy1, A_noisy2)

    assert np.isclose(noise1.var(), noise2.var(), atol=2e-3)
    assert np.isclose(corr_AA, 1, atol=2e-3)
    assert np.isclose(corr_AB, 0, atol=2e-3)
    A_spearman = 1 / np.sqrt(
        (1 + noise1.var() / A1.var()) * (1 + noise2.var() / A2.var())
    )
    assert np.isclose(corr_AA_with_noise, A_spearman, atol=2e-3)


@pytest.mark.parametrize(
    "n_y, n_x, phase_steps, b, c, gamma",
    [(11, 9, 8, -0.05, 1.5, 0.8), (4, 4, 10, -0.05, 1.5, 0.8)],
)
def test_fidelity_phase_calibration_ssa_noise_free(n_y, n_x, phase_steps, b, c, gamma):
    """
    Test computing phase calibration fidelity factor, with the SSA algorithm. Noise-free scenarios.
    """
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    sim = SimulatedWFS(aberrations=aberrations)
    alg = StepwiseSequential(
        feedback=sim, slm=sim.slm, n_x=n_x, n_y=n_y, phase_steps=phase_steps
    )
    result = alg.execute()
    assert result.fidelity_calibration > 0.99

    # SLM with incorrect phase response, noise-free
    linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)
    sim.slm.phase_response = phase_response_test_function(linear_phase, b, c, gamma)
    result = alg.execute()
    assert result.fidelity_calibration < 0.9

    # SLM calibrated with phase response corrected by LUT, noise-free
    sim.slm.lookup_table = lookup_table_test_function(linear_phase, b, c, gamma)
    result = alg.execute()
    assert result.fidelity_calibration > 0.99


@pytest.mark.parametrize(
    "n_y, n_x, phase_steps, gaussian_noise_std", [(4, 4, 10, 0.2), (6, 6, 12, 1.0)]
)
def test_fidelity_phase_calibration_ssa_with_noise(
    n_y, n_x, phase_steps, gaussian_noise_std
):
    """
    Test estimation of phase calibration fidelity factor, with the SSA algorithm. With noise.
    """
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    numerical_aperture = 1.0
    aberration_phase = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
    aberration = StaticSource(aberration_phase, extent=2 * numerical_aperture)
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 250
    src = StaticSource(img, pixel_size = 500 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = SLM(shape=(80, 80))
    sim = Microscope(
        source=src,
        incident_field=slm.field,
        magnification=1,
        numerical_aperture=numerical_aperture,
        aberrations=aberration,
        wavelength=800 * u.nm,
    )
    cam = sim.get_camera(analog_max=1e4, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Define and run WFS algorithm
    alg = StepwiseSequential(
        feedback=roi_detector, slm=slm, n_x=n_x, n_y=n_y, phase_steps=phase_steps
    )
    result_good = alg.execute()
    assert result_good.fidelity_calibration > 0.9

    # SLM with incorrect phase response
    linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)
    slm.phase_response = phase_response_test_function(
        linear_phase, b=0.05, c=0.6, gamma=1.5
    )
    result_good = alg.execute()
    assert result_good.fidelity_calibration < 0.9


@pytest.mark.parametrize(
    "num_blocks, phase_steps, expected_fid, atol", [(10, 8, 1, 1e-6)]
)
def test_measure_modulated_light_dual_phase_stepping_noise_free(
    num_blocks, phase_steps, expected_fid, atol
):
    """Test fidelity estimation due to amount of modulated light. Noise-free."""
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (20, 20))
    sim = SimulatedWFS(aberrations=aberrations)

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light_dual_phase_stepping(
        slm=sim.slm, feedback=sim, phase_steps=phase_steps, num_blocks=num_blocks
    )
    assert np.isclose(fidelity_modulated, expected_fid, atol=atol)


@pytest.mark.parametrize(
    "num_blocks, phase_steps, gaussian_noise_std, atol",
    [(10, 6, 0.0, 1e-6), (6, 8, 2.0, 1e-3)],
)
def test_measure_modulated_light_dual_phase_stepping_with_noise(
    num_blocks, phase_steps, gaussian_noise_std, atol
):
    """Test fidelity estimation due to amount of modulated light. Can test with noise."""
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 100
    src = StaticSource(img, 200 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = SLM(shape=(100, 100))
    sim = Microscope(
        source=src,
        incident_field=slm.field,
        magnification=1,
        numerical_aperture=1.0,
        wavelength=800 * u.nm,
    )
    cam = sim.get_camera(analog_max=1e4, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light_dual_phase_stepping(
        slm=slm, feedback=roi_detector, phase_steps=phase_steps, num_blocks=num_blocks
    )
    assert np.isclose(fidelity_modulated, 1, atol=atol)


@pytest.mark.parametrize(
    "phase_steps, modulated_field_amplitude, non_modulated_field",
    [(6, 1.0, 0.0), (8, 0.5, 0.5), (8, 1.0, 0.25)],
)
def test_measure_modulated_light_noise_free(
    phase_steps, modulated_field_amplitude, non_modulated_field
):
    """Test fidelity estimation due to amount of modulated light. Noise-free."""
    # Perfect SLM, noise-free
    aberrations = np.random.uniform(0.0, 2 * np.pi, (20, 20))
    slm = SLM(
        aberrations.shape,
        field_amplitude=modulated_field_amplitude,
        non_modulated_field_fraction=non_modulated_field,
    )
    sim = SimulatedWFS(aberrations=aberrations, slm=slm)

    # Measure the amount of modulated light (no non-modulated light present)
    fidelity_modulated = measure_modulated_light(
        slm=sim.slm, feedback=sim, phase_steps=phase_steps
    )
    expected_fid = 1.0 / (1.0 + non_modulated_field**2)
    assert np.isclose(fidelity_modulated, expected_fid, rtol=0.1)


@pytest.mark.parametrize(
    "phase_steps, gaussian_noise_std, modulated_field_amplitude, non_modulated_field",
    [(8, 0.0, 0.5, 0.4), (6, 0.0, 1.0, 0.0), (12, 2.0, 1.0, 0.25)],
)
def test_measure_modulated_light_dual_phase_stepping_with_noise(
    phase_steps, gaussian_noise_std, modulated_field_amplitude, non_modulated_field
):
    """Test fidelity estimation due to amount of modulated light. Can test with noise."""
    # === Define mock hardware, perfect SLM ===
    # Aberration and image source
    img = np.zeros((64, 64), dtype=np.int16)
    img[32, 32] = 100
    src = StaticSource(img, pixel_size= 200 * u.nm)

    # SLM, simulation, camera, ROI detector
    slm = SLM(
        shape=(100, 100),
        field_amplitude=modulated_field_amplitude,
        non_modulated_field_fraction=non_modulated_field,
    )
    sim = Microscope(source=src, incident_field=slm.field, wavelength=800 * u.nm)
    cam = sim.get_camera(analog_max=1e3, gaussian_noise_std=gaussian_noise_std)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point

    # Measure the amount of modulated light (no non-modulated light present)
    expected_fid = 1.0 / (1.0 + non_modulated_field**2)
    fidelity_modulated = measure_modulated_light(
        slm=slm, feedback=roi_detector, phase_steps=phase_steps
    )
    assert np.isclose(fidelity_modulated, expected_fid, rtol=0.1)
