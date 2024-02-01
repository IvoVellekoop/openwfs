import numpy as np
import matplotlib.pyplot as plt

from ..openwfs.algorithms.utilities import cnr, signal_std, find_pixel_shift, field_correlation, frame_correlation


def test_signal_std():
    """
    Test signal std, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.rand(200, 200)
    B = np.random.rand(200, 200)
    assert signal_std(A, A) < 1e-6                                      # Test noise only
    assert np.abs(signal_std(A+B, B) - A.std()) < 0.01                  # Test signal+uncorrelated noise
    assert np.abs(signal_std(A+A, A) - np.sqrt(3) * A.std()) < 0.01     # Test signal+correlated noise


def test_cnr():
    """
    Test Contrast to Noise Ratio, corrected for (uncorrelated) noise in the signal.
    """
    A = np.random.randn(300, 300)
    B = np.random.randn(300, 300)
    cnr_gt = 3.0
    assert cnr(A, A) < 1e-6                                                     # Test noise only
    assert np.abs(cnr(cnr_gt*A + B, B) - cnr_gt) < 0.05                         # Test signal+uncorrelated noise
    assert np.abs(cnr(cnr_gt*A + A, A) - np.sqrt((cnr_gt+1)**2 - 1)) < 0.05     # Test signal+correlated noise


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
    a = np.zeros(shape=(2, 3))
    a[1, 2] = 2.0
    a[0, 0] = 0.0

    b = np.ones(shape=(2, 3))
    b[1, 2] = 0.0
    b[0, 0] = 0.0
    assert field_correlation(a, a) == 1.0
    assert field_correlation(2*a, a) == 1.0
    assert field_correlation(a, b) == 0.0
    assert np.abs(field_correlation(a+b, b) - np.sqrt(0.5)) < 1e6


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
    assert np.abs(field_correlation(b, c) - field_correlation(c, b)) < 1e8
    assert np.abs(field_correlation(a+b, b) - np.sqrt(0.5)) < 1e8


def test_frame_correlation():
    """
    Test the frame correlation, i.e. g_2 normalized second order correlation function.
    Test the following:
        g_2 correlation with self (distribution from random.rand) == 1/3
        g_2 correlation with other == 0
    """
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    assert np.abs(frame_correlation(a, a) - 1/3) < 1e-3
    assert np.abs(frame_correlation(a, b)) < 1e-3
