import pytest
import numpy as np
from openwfs.utilities.patterns import tilt, gaussian, disk, propagation, parabolic


@pytest.mark.parametrize("shape", [10, (7, 10)])
def test_tilt(shape):
    t = tilt(shape, (2, 2), (5, 0.71))
    if np.size(shape) == 1:
        shape = (shape, shape)

    # for a default extent of 2.0, 2.0, the tilt should span 4 * t
    phase_diff0 = 5 * 4 * (1.0 - 1.0 / shape[0])
    phase_diff1 = 0.71 * 4 * (1.0 - 1.0 / shape[1])
    assert np.allclose(t[-1, 0] - t[0, 0], phase_diff0)
    assert np.allclose(t[-1, -1] - t[0, -1], phase_diff0)
    assert np.allclose(t[0, -1] - t[0, 0], phase_diff1)
    assert np.allclose(t[-1, -1] - t[-1, 0], phase_diff1)


def test_gaussian_disk_offset():
    shape = (101, 66)
    offset = (0.2, 0.3)
    waist = 0.25
    g = gaussian(shape, extent=(2, 2), waist=waist, offset=offset)
    argmax = np.unravel_index(np.argmax(g), g.shape)

    expected = np.round((np.array(offset) + 1) * (np.array(shape) - 1) / 2.0)
    assert np.allclose(expected, argmax)

    shape = (101, 101)
    d = disk(shape, (2, 2), 2 / 101, offset=offset)
    arg_center = np.argwhere(d > 0.5)[0]
    expected = (np.array(offset) + 1) * (np.array(shape) - 1) / 2.0
    assert np.allclose(expected, arg_center)


def test_parabolic():
    phi = parabolic((11, 11), (2, 2), parabolic_coef=0.5)
    assert np.allclose(phi[5, 5], 0)
    r = -1 + 1 / 11
    assert np.allclose(phi[0, 5], 0.5 * (r**2))
    assert np.allclose(phi[0, 0], 0.5 * (r**2 + r**2))
