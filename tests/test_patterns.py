import pytest
import numpy as np
from openwfs.utilities.patterns import tilt
from openwfs.utilities.patterns import gaussian


@pytest.mark.parametrize("shape", [10, (7, 10)])
def test_tilt(shape):
    t = tilt(shape, (5, 0.71))
    if np.size(shape) == 1:
        shape = (shape, shape)

    # for a default extent of 2.0, 2.0, the tilt should span 4 * t
    phase_diff0 = 5 * 4 * (1.0 - 1.0 / shape[0])
    phase_diff1 = 0.71 * 4 * (1.0 - 1.0 / shape[1])
    assert np.allclose(t[-1, 0] - t[0, 0], phase_diff0)
    assert np.allclose(t[-1, -1] - t[0, -1], phase_diff0)
    assert np.allclose(t[0, -1] - t[0, 0], phase_diff1)
    assert np.allclose(t[-1, -1] - t[-1, 0], phase_diff1)


shape = (101, 101)
offset = (0.2, 0.2)
waist = 0.25
g = gaussian((101, 101), waist=0.25, offset=(0.2, 0.3))
argmax = np.unravel_index(np.argmax(g), g.shape)
print(argmax)
