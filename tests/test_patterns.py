import pytest
import numpy as np
from ..openwfs.slm.patterns import tilt


@pytest.mark.parametrize("resolution", [10, (7, 10)])
def test_tilt(resolution):
    t = tilt(resolution, (5, 0.71))
    if np.size(resolution) == 1:
        resolution = (resolution, resolution)

    phase_diff0 = 5 * 2 * np.pi * (1.0 - 1.0 / resolution[0])
    phase_diff1 = 0.71 * 2 * np.pi * (1.0 - 1.0 / resolution[1])
    assert np.allclose(t[-1, 0] - t[0, 0], phase_diff0)
    assert np.allclose(t[-1, -1] - t[0, -1], phase_diff0)
    assert np.allclose(t[0, -1] - t[0, 0], phase_diff1)
    assert np.allclose(t[-1, -1] - t[-1, 0], phase_diff1)
