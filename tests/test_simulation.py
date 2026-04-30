import logging  # noqa

import astropy.units as u
import numpy as np
import pytest
import skimage

from openwfs.algorithms import StepwiseSequential
from openwfs.processors import SingleRoi
from openwfs.simulation import Microscope, Camera, StaticSource, SLM
from openwfs.utilities import get_pixel_size, Transform
from openwfs.utilities.patterns import tilt, gaussian, parabola, binary_grating, propagation
from openwfs.utilities.tests import get_test_microscope


def test_mock_camera_and_single_roi():
    """
    The MockCamera wraps a Detector producing 2-D data, so that the data can be read by MicroManager.
    The data is converted into a 16-bit int, and methods are added to set width, height, top and bottom.
    By default, the MockCamera scales the input image such that the maximum value is mapped to the unsigned integer
    0xFFFF = (2 ** 16) - 1.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[200, 300] = 39.39  # some random float
    src = Camera(StaticSource(img, pixel_size=450 * u.nm), analog_max=None)
    roi_detector = SingleRoi(src, pos=(200, 300), radius=0)  # Only measure that specific point
    assert roi_detector.read() == int(2**16 - 1)  # it should cast the array into some int


@pytest.mark.parametrize("shape", [(1000, 1000), (999, 999)])
def test_microscope_without_magnification(shape):
    """
    Checks if the microscope can be constructed and read out without exceptions being thrown.
    Also checks if the microscope does not offset the image (for odd and even number of pixels)
    """
    # construct input image
    img = np.zeros(shape, dtype=np.int16)
    img[256, 256] = 100
    src = Camera(StaticSource(img, pixel_size=400 * u.nm), analog_max=0xFFFF)

    # construct microscope
    sim = Microscope(source=src, magnification=1, numerical_aperture=1, wavelength=800 * u.nm)
    cam = Camera(sim, analog_max=None)
    img = cam.read()
    assert img[256, 256] == 2**16 - 1


def test_microscope_and_aberration():
    """
    This test concerns the basic effect of casting an aberration or SLM pattern on the back pupil.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100
    src = Camera(StaticSource(img, pixel_size=400 * u.nm), analog_max=None)

    slm = SLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)

    sim = Microscope(
        source=src,
        magnification=1,
        incident_field=slm.field,
        numerical_aperture=1,
        wavelength=800 * u.nm,
    )

    without_aberration = sim.read()[256, 256]
    slm.set_phases(aberrations)
    with_aberration = sim.read()[256, 256]
    assert with_aberration < without_aberration


def test_slm_and_aberration():
    """
    As mentioned in the previous test, casting a pattern on the pupil plane with an SLM and an aberration
    should produce the same effect. We will test that here by projecting two opposing patterns on the pupil plane.
    (Which should do nothing in the image plane)
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100
    src = Camera(StaticSource(img, pixel_size=400 * u.nm), analog_max=None)

    slm = SLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0) * 0
    slm.set_phases(-aberrations)
    aberration = StaticSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)

    sim1 = Microscope(
        source=src,
        incident_field=slm.field,
        numerical_aperture=1.0,
        aberrations=aberration,
        wavelength=800 * u.nm,
    )
    sim2 = Microscope(source=src, numerical_aperture=1.0, wavelength=800 * u.nm)

    # We correlate the two.
    # Any discrepancy between the two matrices should throw an error
    # try putting one of the wavelengths to 800

    a = sim1.read()
    b = sim2.read()
    norm_a = a / np.linalg.norm(a[:])
    norm_b = b / np.linalg.norm(b[:])

    assert abs(np.vdot(norm_a, norm_b)) >= 1


def shift_between_img_max(ref_img, shifted_img):
    return np.subtract(
        np.unravel_index(np.argmax(ref_img), ref_img.shape), np.unravel_index(np.argmax(shifted_img), shifted_img.shape)
    )


def test_slm_tilt():
    """
    Display a tilt on the SLM should result in an image plane shift. If the magnification is 1, this should
    correspond to a tilt of 1 pixel for a 2 pi phase shift.
    """
    mic, slm, src = get_test_microscope(mic_args={"numerical_aperture": 0.5})

    ref_img = mic.read()
    shift = np.multiply(get_pixel_size(ref_img), (10, -5))
    gradient = shift * np.pi * mic.numerical_aperture / mic.wavelength
    tilt_pattern = tilt(slm.pixels.data_shape, extent=(2, 2), g=gradient)
    slm.set_phases(tilt_pattern)

    shifted_img = mic.read()
    measured_shift = shift_between_img_max(ref_img, shifted_img)
    assert np.allclose(np.multiply(get_pixel_size(ref_img), measured_shift), shift)


def test_microscope_wavefront_shaping(caplog):
    """
    Reproduces a bug that occurs due to the location of the measurements.wait() command.
    """
    # caplog.set_level(logging.DEBUG)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi

    aberration = StaticSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)  # note: incorrect scaling!

    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100

    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (250, 200)

    img[signal_location] = 100
    src = StaticSource(img, pixel_size=400 * u.nm)

    slm = SLM(shape=(1000, 1000))

    sim = Microscope(
        source=src,
        incident_field=slm.field,
        numerical_aperture=1,
        aberrations=aberration,
        wavelength=800 * u.nm,
    )

    cam = Camera(sim, analog_max=100)
    roi_detector = SingleRoi(cam, pos=signal_location, radius=0)  # Only measure that specific point

    alg = StepwiseSequential(feedback=roi_detector, slm=slm, phase_steps=3, n_x=3, n_y=3)
    t = alg.execute().t

    # test if the modes differ. The error causes them not to differ
    assert np.std(t[:][1:]) > 0


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


def test_mock_slm_lut_and_phase_response():
    """
    Test the lookup table and phase response of the MockSLM.
    """
    # === Test default lookup table and phase response ===
    # Includes edge cases like rounding/wrapping: -0.501 -> 255, -0.499 -> 0
    input_phases_a = (
        np.asarray(
            (
                -1,
                -0.501,
                -0.499,
                0,
                1,
                64,
                128,
                192,
                255,
                255.499,
                255.501,
                256,
                257,
                511,
                512,
            )
        )
        * 2
        * np.pi
        / 256
    )
    expected_output_phases_a = (
        np.asarray((255, 255, 0, 0, 1, 64, 128, 192, 255, 255, 0, 0, 1, 255, 0)) * 2 * np.pi / 256
    )
    slm1 = SLM(shape=(3, input_phases_a.shape[0]))
    slm1.set_phases(input_phases_a)
    assert np.all(np.abs(slm1.phases.read() - expected_output_phases_a) < 1e6)

    # === Test setting custom phase response of SLM ===
    # Construct custom phase response
    b = -0.02
    c = 1.2
    gamma = 1.6
    linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)
    phase_response = phase_response_test_function(linear_phase, b, c, gamma)

    # Apply and test custom synthetic phase response
    slm2 = SLM(shape=(3, 256))
    slm2.phase_response = phase_response
    slm2.set_phases(linear_phase)
    assert np.all(np.abs(slm2.phases.read() - phase_response) < 1e-6)

    # === Test custom lookup table ===
    lookup_table = lookup_table_test_function(linear_phase, b, c, gamma)
    slm3 = SLM(shape=(3, 256))
    slm3.lookup_table = lookup_table
    slm3.set_phases(linear_phase)
    assert np.all(
        np.abs(slm3.phases.read() - inverse_phase_response_test_function(linear_phase, b, c, gamma))
        < (1.1 * np.pi / 256)
    )

    # === Test custom lookup table that counters custom synthetic phase response ===
    linear_phase_highres = np.arange(0, 2 * np.pi * 255.49 / 256, 0.25 * 2 * np.pi / 256)
    slm4 = SLM(shape=(3, linear_phase_highres.shape[0]))
    slm4.phase_response = phase_response
    slm4.lookup_table = lookup_table
    slm4.set_phases(linear_phase_highres)
    assert np.all(np.abs(slm4.phases.read()[0] - linear_phase_highres) < (3 * np.pi / 256))


@pytest.mark.parametrize("extent", [2, 4])
def test_parabola_shift(extent):
    """
    Tests that a display of a parabola pattern on the SLM with an certain offset results in the expected shift in the image plane.
    """
    offset_focal = np.array([1100, 400]) * u.nm
    coef_parabola = 0.1

    mic, slm, src = get_test_microscope()

    # Pupil shift required to shift the image by offset_focal. See docs of parabola
    pupil_offset = (
        np.multiply(np.multiply(offset_focal, mic.numerical_aperture), -np.pi / mic.wavelength) / coef_parabola
    )
    phi = parabola((1024, 1024), extent, coef_parabola)
    slm.set_phases(phi)
    mic.slm_transform = Transform(np.eye(2) * extent / 2, np.zeros(2), np.zeros(2))
    img_1 = mic.read()

    phi = parabola((1024, 1024), extent, coef_parabola, offset=pupil_offset)
    slm.set_phases(phi)
    img_2 = mic.read()
    measured_shift = shift_between_img_max(img_1, img_2) * get_pixel_size(img_1)
    assert np.allclose(measured_shift, offset_focal)


@pytest.mark.parametrize("extent", [2, 4])
def test_parabola(extent):
    # Test that the parabola pattern produces the expected shift in the image plane if the parabola is not centered on the back pupil plane of the image.
    na = 0.9
    wav = 500 * u.nm
    mic, slm, src = get_test_microscope(
        mic_args={"numerical_aperture": na, "wavelength": wav}, src_args={"pixel_size": 50 * u.nm}
    )
    extent = 6
    desired_shift = 1000 * u.nm
    periodicity = mic.wavelength / desired_shift / mic.numerical_aperture

    phi = binary_grating((512, 512), extent, periodicity, (0, np.pi), angle=45 * u.deg)

    mic.slm_transform = Transform(np.eye(2) * extent / 2, np.zeros(2), np.zeros(2))
    slm.set_phases(0)
    img_ref = mic.read()
    slm.set_phases(phi)
    img = mic.read()

    assert np.allclose(
        np.abs(shift_between_img_max(img_ref, img) * get_pixel_size(img_ref)),
        np.ones(2) * np.sqrt(2) / 2 * desired_shift.to(u.nm),
        rtol=0.05,
    )


@pytest.mark.parametrize("n", [1, 2])
def test_propagation(n):
    # Test that a SLM propagation pattern can compensate for the defocus of the microscope
    mic, slm, src = get_test_microscope(mic_args={"immersion_refractive_index": n})
    img_ref = mic.read()
    phi = propagation(512, 2, 10 * u.um, mic.wavelength, n, mic.numerical_aperture)
    slm.set_phases(phi)
    mic.z_stage.position = -10 * u.um
    img = mic.read()

    assert np.allclose(img, img_ref, atol=1e-3)


def test_non_linear_microscope():
    mic_1, slm_1, src_1 = get_test_microscope()
    img_ref = mic_1.read()

    mic_2, slm_2, src_2 = get_test_microscope(mic_args={"nonlinearity": 2})
    img_2p = mic_2.read()

    assert np.allclose(img_2p, img_ref**2)


def test_microscope_z_stack():
    mic, slm, src = get_test_microscope()
    z = [10, -10] * u.um
    imgs = mic.z_stack_read(z)

    mic.z_stage.position = z[0]
    img_z_first = mic.read()

    mic.z_stage.position = z[1]
    img_z_second = mic.read()

    assert np.allclose(imgs[:, :, 0], img_z_first)
    assert np.allclose(imgs[:, :, 1], img_z_second)
    assert imgs.shape == (img_ref.shape[0], img_ref.shape[1], z.size)
