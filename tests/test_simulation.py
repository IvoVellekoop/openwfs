import logging  # noqa
import pytest
import numpy as np
from ..openwfs.algorithms import StepwiseSequential
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import Microscope, MockCamera, MockSource, MockSLM
from ..openwfs.utilities.patterns import tilt
import skimage
import astropy.units as u


def test_mock_camera_and_single_roi():
    """
    The MockCamera wraps a Detector producing 2-D data, so that the data can be read by MicroManager.
    The data is converted into a 16-bit int, and methods are added to set width, height, top and bottom.
    By default, the MockCamera scales the input image such that the maximum value is mapped to the unsigned integer
    0xFFFF = (2 ** 16) - 1.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[200, 300] = 39.39  # some random float
    src = MockCamera(MockSource(img, 450 * u.nm))
    roi_detector = SingleRoi(src, pos=(200, 300), radius=0)  # Only measure that specific point
    assert roi_detector.read() == int(2 ** 16 - 1)  # it should cast the array into some int


@pytest.mark.parametrize("shape", [(1000, 1000), (999, 999)])
def test_microscope_without_magnification(shape):
    """
    Checks if the microscope can be constructed and read out without exceptions being thrown.
    Also checks if the microscope does not offset the image (for odd and even number of pixels)
    """
    # construct input image
    img = np.zeros(shape, dtype=np.int16)
    img[256, 256] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    # construct microscope
    sim = Microscope(source=src, magnification=1, numerical_aperture=1, wavelength=800 * u.nm)

    cam = sim.get_camera()
    img = cam.read()
    assert img[256, 256] == 2 ** 16 - 1


def test_microscope_and_aberration():
    """
    This test concerns the basic effect of casting an aberration or SLM pattern on the back pupil.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    slm = MockSLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)

    sim = Microscope(source=src, magnification=1, slm=slm.pixels(), numerical_aperture=1, wavelength=800 * u.nm)

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
    src = MockCamera(MockSource(img, 400 * u.nm))

    slm = MockSLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0) * 0
    slm.set_phases(-aberrations)
    aberration = MockSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)

    sim1 = Microscope(source=src, slm=slm.pixels(), numerical_aperture=1.0, aberrations=aberration,
                      wavelength=800 * u.nm)
    sim2 = Microscope(source=src, numerical_aperture=1.0, wavelength=800 * u.nm)

    # We correlate the two.
    # Any discrepancy between the two matrices should throw an error
    # try putting one of the wavelengths to 800

    a = sim1.read()
    b = sim2.read()
    norm_a = a / np.linalg.norm(a[:])
    norm_b = b / np.linalg.norm(b[:])

    assert abs(np.vdot(norm_a, norm_b)) >= 1


def test_slm_tilt():
    """
    Display a tilt on the SLM should result in an image plane shift. If the magnification is 1, this should
    correspond to a tilt of 1 pixel for a 2 pi phase shift.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (256, 250)
    img[signal_location] = 100
    pixel_size = 400 * u.nm
    wavelength = 750 * u.nm
    src = MockCamera(MockSource(img, pixel_size))

    slm = MockSLM(shape=(1000, 1000))

    na = 1.0
    sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=na, wavelength=wavelength)

    # introduce a tilted pupil plane
    # the input parameter to `tilt` corresponds to a shift 2.0/π the Abbe diffraction limit.
    shift = np.array((-24, 40))
    step = wavelength / (np.pi * na)
    slm.set_phases(tilt(1000, - shift * pixel_size / step))

    new_location = signal_location + shift

    cam = sim.get_camera()
    img = cam.read(immediate=True)
    max_pos = np.unravel_index(np.argmax(img), img.shape)
    assert np.all(max_pos == new_location)


def test_microscope_wavefront_shaping(caplog):
    """
    Reproduces a bug that occurs due to the location of the measurements.wait() command.
    """
    # caplog.set_level(logging.DEBUG)
    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi

    aberration = MockSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)

    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100

    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (250, 200)

    img[signal_location] = 100
    src = MockSource(img, 400 * u.nm)

    slm = MockSLM(shape=(1000, 1000))

    sim = Microscope(source=src, slm=slm.pixels(), numerical_aperture=1, aberrations=aberration, wavelength=800 * u.nm)

    cam = sim.get_camera(analog_max=100)
    roi_detector = SingleRoi(cam, pos=signal_location, radius=0)  # Only measure that specific point

    alg = StepwiseSequential(feedback=roi_detector, slm=slm, phase_steps=3, n_x=3, n_y=3)
    t = alg.execute().t

    # test if the modes differ. The error causes them not to differ
    assert np.std(t[:][1:]) > 0
