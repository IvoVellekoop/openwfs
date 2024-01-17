import numpy as np
import matplotlib.pyplot as plt

from ..openwfs.algorithms.utilities import cnr, signal_std, find_pixel_shift


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
