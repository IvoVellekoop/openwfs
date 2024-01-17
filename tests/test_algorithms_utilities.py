import numpy as np
import matplotlib.pyplot as plt

from ..openwfs.algorithms.utilities import find_pixel_shift


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
