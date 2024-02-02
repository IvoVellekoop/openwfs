import numpy as np
import skimage
import astropy.units as u
import pytest

from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import Microscope, MockSource, MockSLM
from ..openwfs.algorithms import FourierDualReference
from ..openwfs.algorithms.utilities import WFSController
from ..openwfs.algorithms.troubleshoot import cnr, signal_std, find_pixel_shift, field_correlation, frame_correlation


def test_signal_std():
    """
    Test signal std, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.rand(400, 400)
    B = np.random.rand(400, 400)
    assert signal_std(A, A) < 1e-6                                      # Test noise only
    assert np.abs(signal_std(A+B, B) - A.std()) < 0.005                 # Test signal+uncorrelated noise
    assert np.abs(signal_std(A+A, A) - np.sqrt(3) * A.std()) < 0.005    # Test signal+correlated noise


def test_cnr():
    """
    Test Contrast to Noise Ratio, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.randn(500, 500)
    B = np.random.randn(500, 500)
    cnr_gt = 3.0                                                                # Ground Truth
    assert cnr(A, A) < 1e-6                                                     # Test noise only
    assert np.abs(cnr(cnr_gt*A + B, B) - cnr_gt) < 0.01                         # Test signal+uncorrelated noise
    assert np.abs(cnr(cnr_gt*A + A, A) - np.sqrt((cnr_gt+1)**2 - 1)) < 0.01     # Test signal+correlated noise


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
    Test the following:
        self-correlation == 1
        invariant under scalar multiplication
        correlation with orthogonal array == 0
        correlation with arguments swapped
        correlation with self + orthogonal array
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

    assert field_correlation(a, a) == 1.0
    assert field_correlation(2*a, a) == 1.0
    assert field_correlation(a, b) == 0.0
    assert np.abs(field_correlation(b, c) - np.conj(field_correlation(c, b))) < 1e-10
    assert np.abs(field_correlation(a+b, b) - np.sqrt(0.5)) < 1e-10


def test_frame_correlation():
    """
    Test the frame correlation, i.e. g_2 normalized second order correlation function.
    Test the following:
        g_2 correlation with self == 1/3 for distribution from random.rand
        g_2 correlation with other == 0
    """
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    assert np.abs(frame_correlation(a, a) - 1/3) < 2e-3
    assert np.abs(frame_correlation(a, b)) < 2e-3


@pytest.mark.skip(reason="This test does not test anything yet and gives a popop graph.")
def test_wfs_troubleshooter():
    # Define mock hardware
    numerical_aperture = 1.0
    aberration_phase = 0.5 * skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
    aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)

    img = np.zeros((256, 256), dtype=np.int16)
    img[128, 128] = 95
    img[70, 70] = 50
    img[40, 40] = 50
    img[70, 40] = 40
    img[40, 70] = 30
    img[128, 70] = 80

    src = MockSource(img, 400 * u.nm)

    slm_shape = (1000, 1000)
    slm = MockSLM(shape=slm_shape)

    sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=numerical_aperture,
                     aberrations=aberration, wavelength=800 * u.nm)

    cam = sim.get_camera(analog_max=100.0, gaussian_noise_std=0.000)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point
    alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=slm_shape,
                               k_angles_min=-1, k_angles_max=1, phase_steps=3)
    control = WFSController(alg, cam)

    control.troubleshoot()

    # #=== Uncomment for debugging ===#
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.log10(control.read_after_frame_flatwf().clip(1)), vmin=0, vmax=5)
    # plt.title('Flat wavefront')
    # plt.colorbar()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.log10(control.read_after_frame_shapedwf().clip(1)), vmin=0, vmax=5)
    # plt.title(f'Shaped wavefront\nCNR: {control.frame_cnr:.3f}, η_σ: {control.contrast_enhancement:.3f}')
    # plt.colorbar()
    # plt.show()
