import pytest
import numpy as np
from ..openwfs.slm.patterns import tilt


@pytest.mark.parametrize("shape", [10, (7, 10)])
def test_tilt(shape):
    t = tilt(shape, (5, 0.71))
    if np.size(shape) == 1:
        shape = (shape, shape)

    phase_diff0 = 5 * 2 * (1.0 - 1.0 / shape[0])
    phase_diff1 = 0.71 * 2 * (1.0 - 1.0 / shape[1])
    assert np.allclose(t[-1, 0] - t[0, 0], phase_diff0)
    assert np.allclose(t[-1, -1] - t[0, -1], phase_diff0)
    assert np.allclose(t[0, -1] - t[0, 0], phase_diff1)
    assert np.allclose(t[-1, -1] - t[-1, 0], phase_diff1)
