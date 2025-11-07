from typing import Optional

import astropy.units as u
import numpy as np
import pytest
import skimage
from scipy.linalg import hadamard
from scipy.ndimage import zoom

from . import complex_random
from openwfs.algorithms import (
    StepwiseSequential,
    FourierDualReference,
    DualReference,
    SimpleGenetic,
)
from openwfs.algorithms.troubleshoot import field_correlation
from openwfs.algorithms.utilities import WFSController
from openwfs.processors import SingleRoi
from openwfs.simulation import SimulatedWFS, StaticSource, SLM, Microscope
from openwfs.simulation.mockdevices import GaussianNoise, Camera


@pytest.fixture(autouse=True)
def reset():
    np.random.seed(42)  # for reproducibility


@pytest.mark.parametrize("algorithm", ["ssa", "fourier"])
@pytest.mark.parametrize("shape, feedback_shape", [((4, 7), (210,)), ((10, 7), (21, 10)), ((20, 31), (3, 7, 10))])
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_multi_target_algorithms(shape: tuple[int, int], feedback_shape: tuple[int, ...], noise: float, algorithm: str):
    """
    Test the multi-target capable algorithms (SSA and Fourier dual ref).

    This tests checks if the algorithm achieves the theoretical enhancement,
    and it also verifies that the enhancement and noise fidelity
    are estimated correctly by the algorithm.
    """
    M = np.prod(feedback_shape)  # number of targets
    phase_steps = 6

    # create feedback object, with noise if needed
    sim = SimulatedWFS(t=complex_random((*feedback_shape, *shape)))
    sim.slm.set_phases(0.0)
    I_0 = np.mean(sim.read())
    feedback = GaussianNoise(sim, std=I_0 * noise)

    if algorithm == "ssa":
        alg = StepwiseSequential(
            feedback=feedback,
            slm=sim.slm,
            n_x=shape[1],
            n_y=shape[0],
            phase_steps=phase_steps,
        )
        N = np.prod(shape)  # number of input modes
        # SSA is inaccurate if N is slightly lower because the reference beam varies with each segment.
        # The variation is of order 1/N in the signal, so the fidelity is (N-1)/N.
        # todo: check!
        alg_fidelity = (N - 1) / N
        signal = (N - 1) / N**2  # for estimating SNR
        masks = [True]  # use all segments for determining fidelity
    else:  # 'fourier':
        alg = FourierDualReference(
            feedback=feedback,
            slm=sim.slm,
            slm_shape=shape,
            k_radius=(np.min(shape) - 1) // 2,
            phase_steps=phase_steps,
        )
        N = len(alg.phase_patterns[0]) + len(alg.phase_patterns[1])  # number of input modes
        alg_fidelity = 1.0  # Fourier is accurate for any N
        signal = 1 / 2  # for estimating SNR.
        masks = alg.masks  # use separate halves of the segments for determining fidelity

    # Execute the algorithm to get the optimized wavefront
    # for all targets simultaneously
    result = alg.execute()

    # Determine the optimized intensities in each of the targets individually
    # Also estimate the fidelity of the transmission matrix reconstruction
    # This fidelity is determined row by row, since we need to compensate
    # the unknown phases. The normalization of the correlation function
    # is performed on all rows together, not per row, to increase
    # the accuracy of the estimate.
    # For the dual reference algorithm, the left and right half will have a different overall amplitude factor.
    # This is corrected for by computing the correlations for the left and right half separately
    #
    I_opt = np.zeros(feedback_shape)
    for b in np.ndindex(feedback_shape):
        sim.slm.set_phases(-np.angle(result.t[b]))
        I_opt[b] = feedback.read()[b]

    t_correlation = t_fidelity(result.t, sim.t, masks=masks)

    # Check the enhancement, noise fidelity and
    # the fidelity of the transmission matrix reconstruction
    coverage = N / np.prod(shape)
    print(signal)
    print(noise)
    theoretical_noise_fidelity = signal * phase_steps / (signal * phase_steps + noise**2)

    enhancement = I_opt.mean() / I_0
    theoretical_enhancement = np.pi / 4 * theoretical_noise_fidelity * alg_fidelity * (N - 1) + 1
    estimated_enhancement = result.estimated_enhancement.mean() * alg_fidelity
    theoretical_t_correlation = theoretical_noise_fidelity * alg_fidelity * coverage
    estimated_t_correlation = result.fidelity_noise * result.fidelity_calibration * alg_fidelity * coverage
    tolerance = 4.0 / np.sqrt(M)  # TODO: find out if this should be stricter
    print(
        f"\nenhancement:      \ttheoretical= {theoretical_enhancement},\testimated={estimated_enhancement},\tactual: {enhancement}"
    )
    print(
        f"t-matrix fidelity:\ttheoretical = {theoretical_t_correlation},\testimated = {estimated_t_correlation},\tactual = {t_correlation}"
    )
    print(f"noise fidelity:   \ttheoretical = {theoretical_noise_fidelity},\testimated = {result.fidelity_noise}")
    print(f"comparing at relative tolerance: {tolerance}")

    assert np.allclose(
        enhancement, theoretical_enhancement, rtol=tolerance
    ), f"""
        The SSA algorithm did not enhance the focus as much as expected.
        Theoretical {theoretical_enhancement}, got {enhancement}"""

    assert np.allclose(
        estimated_enhancement, enhancement, rtol=tolerance
    ), f"""
         The SSA algorithm did not estimate the enhancement correctly.
         Estimated {estimated_enhancement}, got {enhancement}"""

    assert np.allclose(
        t_correlation, theoretical_t_correlation, rtol=tolerance
    ), f"""
        The SSA algorithm did not measure the transmission matrix correctly.
        Expected {theoretical_t_correlation}, got {t_correlation}"""

    assert np.allclose(
        estimated_t_correlation, theoretical_t_correlation, rtol=tolerance
    ), f"""
        The SSA algorithm did not estimate the fidelity of the transmission matrix correctly.
        Expected {theoretical_t_correlation}, got {estimated_t_correlation}"""

    assert np.allclose(
        result.fidelity_noise, theoretical_noise_fidelity, rtol=tolerance
    ), f"""
        The SSA algorithm did not estimate the noise correctly.
        Expected {theoretical_noise_fidelity}, got {result.fidelity_noise}"""


def half_mask(shape):
    """
    Args:
        shape: shape of the output array

    Returns:
        Returns a boolean mask with [:, :shape[2]] set to False
        and [:, shape[1]] set to True].
    """
    mask = np.zeros(shape, dtype=bool)
    mask[:, shape[1] // 2 :] = True
    return mask


def t_fidelity(
    t: np.ndarray, t_correct: np.ndarray, *, columns: int = 2, masks: Optional[tuple[np.ndarray, ...]] = (True,)
) -> float:
    """
    Compute the fidelity of the measured transmission matrix.

    Since the overall phase for each row is unknown, the fidelity is computed row by row.
    Moreover, for dual-reference algorithms the left and right half of the wavefront
    may have different overall amplitude factors. This is corrected for by computing
    the fidelity for the left and right half separately.

    The fidelities for all rows (and halves) are weighted by the 'intensity' in the correct transmission matrix.

    Args:
        t: The measured transmission matrix. 'rows' of t are the *last* index
        t_correct: The correct transmission matrix
        columns: The number of columns in the transmission matrix
        masks: Masks for the left and right half of the wavefront, or None to use the full wavefront
    """
    fidelity = 0.0
    norm = 0.0
    for r in np.ndindex(t.shape[:-columns]):  # each row
        for m in masks:
            rm = t[r][m]
            rm_c = t_correct[r][m]
            fidelity += abs(np.vdot(rm, rm_c) ** 2 / np.vdot(rm, rm))
            norm += np.vdot(rm_c, rm_c)

    return fidelity / norm


def test_fourier2():
    """Test the Fourier dual reference algorithm using WFSController."""
    slm_shape = (10, 10)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)
    sim = SimulatedWFS(aberrations=aberrations)
    alg = WFSController(
        FourierDualReference, feedback=sim, slm=sim.slm, slm_shape=slm_shape, k_radius=3.5, phase_steps=3
    )

    # check if the attributes of the algorithm were passed through correctly
    assert alg.k_radius == 3.5
    alg.k_radius = 2.5
    assert alg.k_radius == 2.5
    before = sim.read()
    alg.wavefront = WFSController.State.OPTIMIZED  # this will trigger the algorithm to optimize the wavefront
    after = sim.read()
    alg.wavefront = WFSController.State.FLAT  # this set the wavefront back to flat
    before2 = sim.read()
    assert before == before2
    assert after / before > 3.0


@pytest.mark.skip("Not implemented")
def test_fourier_microscope():
    aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
    aberration = StaticSource(aberration_phase, pixel_size=2.0 / np.array(aberration_phase.shape))
    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (250, 250)
    img[signal_location] = 100
    slm_shape = (1000, 1000)

    src = StaticSource(img, pixel_size=400 * u.nm)
    slm = SLM(shape=(1000, 1000))
    sim = Microscope(
        source=src,
        incident_field=slm.field,
        magnification=1,
        numerical_aperture=1,
        aberrations=aberration,
        wavelength=800 * u.nm,
    )
    cam = Camera(sim, analog_max=100)
    roi_detector = SingleRoi(cam, pos=(250, 250))  # Only measure that specific point
    alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=slm_shape, k_radius=1.5, phase_steps=3)
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
    alg = FourierDualReference(
        feedback=sim,
        slm=sim.slm,
        slm_shape=np.shape(aberrations),
        k_radius=3.0,
        phase_steps=3,
    )
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
    alg = FourierDualReference(
        feedback=sim,
        slm=sim.slm,
        slm_shape=np.shape(aberrations),
        k_radius=1.5,
        phase_steps=3,
    )
    t = alg.execute().t

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t)
    sim.slm.set_phases(0)

    optimised_wf -= 5
    signals = []

    for n in range(5):
        optimised_wf += 2
        sim.slm.set_phases(optimised_wf)
        signal = sim.read()
        signals.append(signal)

    assert (
        np.std(signals) / np.mean(signals) < 0.001
    ), f"""The response of SimulatedWFS is sensitive to a flat 
        phase shift. This is incorrect behaviour"""


@pytest.mark.parametrize("optimized_reference", [True, False])
@pytest.mark.parametrize("type", ["flat", "phase_step", "amplitude_step"])
def test_flat_wf_response_fourier(optimized_reference, type):
    """
    Test the response of the Fourier-based WFS method when the solution is flat
    A flat solution means that the optimal correction is no correction.
    Also tests if stitching is done correctly by having an aberration pattern which is flat (but different) on the two halves.

    test the optimized wavefront by checking if it has irregularities.
    """
    t = np.ones(shape=(4, 4), dtype=np.complex64)
    if type == "phase_step":
        t[:, 2:] = np.exp(2.0j)
    elif type == "amplitude_step":
        t[:, 2:] = 2.0

    sim = SimulatedWFS(t=t)

    alg = FourierDualReference(
        feedback=sim,
        slm=sim.slm,
        slm_shape=np.shape(t),
        k_radius=1.5,
        phase_steps=3,
        optimized_reference=optimized_reference,
    )

    result = alg.execute()
    assert (
        abs(field_correlation(result.t / np.abs(result.t), t / np.abs(t))) > 0.99
    ), "The phases were not calculated correctly"
    assert t_fidelity(result.t, t, masks=alg.masks) > 0.99, "The amplitudes were not calculated correctly"


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
    aberrations = np.random.uniform(0.0, 2 * np.pi, (5, 2, 256, 256))
    sim = SimulatedWFS(aberrations=aberrations)

    alg = StepwiseSequential(feedback=sim, slm=sim.slm)
    t = alg.execute().t

    # compute the phase pattern to optimize the intensity in target 2,1
    target = (2, 1)
    optimised_wf = -np.angle(t[(*target, ...)])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert (
        enhancement[target] >= 3.0
    ), f"""The SSA algorithm did not enhance focus as much as expected.
            Expected at least 3.0, got {enhancement}"""


def test_multidimensional_feedback_fourier():
    aberrations = np.random.uniform(0.0, 2 * np.pi, (5, 2, 256, 256))
    sim = SimulatedWFS(aberrations=aberrations)

    # input the camera as a feedback object, such that it is multidimensional
    alg = FourierDualReference(feedback=sim, slm=sim.slm, k_radius=3.5, phase_steps=3)
    t = alg.execute().t

    # compute the phase pattern to optimize the intensity in target 0
    optimised_wf = -np.angle(t[2, 1, :, :])

    # Calculate the enhancement factor
    # Note: technically this is not the enhancement, just the ratio after/before
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(optimised_wf)
    after = sim.read()
    enhancement = after / before

    assert (
        enhancement[2, 1] >= 3.0
    ), f"""The algorithm did not enhance the focus as much as expected.
            Expected at least 3.0, got {enhancement}"""


@pytest.mark.parametrize("population_size, elite_size", [(30, 15)])  # , (30, 5)])
def test_simple_genetic(population_size: int, elite_size: int):
    """
    Test the SimpleGenetic algorithm.
    Note: this is not very rigid test, as we currently don't have theoretical expectations for the performance.
    """
    shape = (100, 71)
    sim = SimulatedWFS(t=complex_random(shape), multi_threaded=False)
    alg = SimpleGenetic(
        feedback=sim,
        slm=sim.slm,
        shape=shape,
        population_size=population_size,
        elite_size=elite_size,
        generations=1000,
    )
    result = alg.execute()
    sim.slm.set_phases(0.0)
    before = sim.read()
    sim.slm.set_phases(-np.angle(result.t))
    after = sim.read()

    print(after / before)
    assert after / before > 4


@pytest.mark.parametrize("basis_str", ("plane_wave", "hadamard"))
@pytest.mark.parametrize("shape", ((8, 8), (16, 4)))
def test_dual_reference_ortho_split(basis_str: str, shape: tuple[int, int]):
    """Test dual reference in iterative mode with an orthonormal phase-only basis.
    Two types of bases are tested: plane waves and Hadamard"""
    N = shape[0] * (shape[1] // 2)
    modes_shape = (N, shape[0], shape[1] // 2)
    if basis_str == "plane_wave":
        # Create a full plane wave basis for one half of the SLM.
        phases = np.angle(np.fft.fft2(np.eye(N).reshape(modes_shape), axes=(1, 2)))
    elif basis_str == "hadamard":
        phases = np.angle(hadamard(N).reshape(modes_shape))
    else:
        raise f'Unknown type of basis "{basis_str}".'

    mask = half_mask(shape)
    phases_set = np.pad(phases, ((0, 0), (0, 0), (0, shape[1] // 2)))

    sim = SimulatedWFS(t=complex_random(shape))

    alg = DualReference(
        feedback=sim,
        slm=sim.slm,
        phase_patterns=(phases_set, np.flip(phases_set, axis=2)),
        group_mask=mask,
        iterations=4,
    )

    # Checks for orthonormal basis properties
    assert np.allclose(alg.gram[0], np.eye(N), atol=1e-6)  # Gram matrix must be I

    # Cobasis vectors are just the complex conjugates
    assert np.allclose(alg.cobasis[0], np.exp(-1j * phases_set) * abs(alg.cobasis[0]), atol=1e-6)

    # Test phase-only field correlation
    result = alg.execute()
    sim_t_phase_only = np.exp(1j * np.angle(sim.t))
    result_t_phase_only = np.exp(1j * np.angle(result.t))
    assert np.abs(field_correlation(sim_t_phase_only, result_t_phase_only)) > 0.999

    assert t_fidelity(result.t, sim.t, masks=alg.masks) > 0.9


def test_dual_reference_non_ortho_split():
    """
    Test dual reference with a non-orthogonal basis.
    """
    # Create set of modes that are barely linearly independent
    N1 = 6
    N2 = 3
    M = N1 * N2
    mode_set_half = np.exp(2j * np.pi / 3 * np.eye(M).reshape((N2, N1, M))).T / np.sqrt(M)

    # note: typically we just use the other half for the mode set B, but here we set the half to 0 to
    # make sure it is not used.
    mode_set = np.pad(mode_set_half, ((0, 0), (0, 0), (0, N2)))
    phases_set = np.angle(mode_set)
    mask = half_mask((N1, 2 * N2))

    # Create aberrations
    x = np.linspace(-1, 1, N1).reshape((1, -1))
    y = np.linspace(-1, 1, N1).reshape((-1, 1))
    aberrations = (np.sin(0.8 * np.pi * x) * np.cos(1.3 * np.pi * y) * (0.8 * np.pi + 0.4 * x + 0.4 * y)) % (2 * np.pi)
    aberrations[0:1, :] = 0
    aberrations[:, 0:2] = 0

    sim = SimulatedWFS(aberrations=aberrations)

    alg = DualReference(
        feedback=sim,
        slm=sim.slm,
        phase_patterns=(phases_set, np.flip(phases_set, axis=2)),
        group_mask=mask,
        phase_steps=4,
        iterations=4,
    )

    result = alg.execute()

    aberration_field = np.exp(1j * aberrations)
    t_field = np.exp(1j * np.angle(result.t))

    assert np.abs(field_correlation(aberration_field, t_field)) > 0.999
