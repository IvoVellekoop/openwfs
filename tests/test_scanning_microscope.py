import numpy as np

from ..openwfs.devices import ScanningMicroscope
from ..openwfs.slm.patterns import coordinate_range
from ..openwfs.utilities import imshow
import astropy.units as u
import pytest


@pytest.mark.parametrize("direction", ['horizontal', 'vertical'])
def test_scan_pattern(direction):
    shape = (100, 80)
    padding = 0.0
    scale = 440 * u.um / u.V
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev4/ao0', -2.0 * u.V, 2.0 * u.V),
                                 axis1=('Dev4/ao1', -2.0 * u.V, 2.0 * u.V), input=('Dev4/ai0', -1.0 * u.V, 1.0 * u.V),
                                 data_shape=shape, scale=scale, simulation=direction, padding=padding)

    assert np.allclose(scanner.extent, scale * 4.0 * u.V)

    # check if returned pattern is correct
    (y, x) = coordinate_range(shape, 4.0 * u.V)
    full = scanner.read().astype('float32') - 0x8000
    if direction == 'horizontal':
        assert np.allclose(x.to_value(u.mV), full, atol=0.5)
    else:
        assert np.allclose(y.to_value(u.mV), full, atol=0.5)

    # test setting the ROI
    left = 10
    top = 30
    width = 29
    height = 17
    scanner.left = left
    scanner.top = top
    scanner.width = width
    scanner.height = height

    assert scanner.left == left
    assert scanner.top == top
    assert scanner.width == width
    assert scanner.height == height
    assert scanner.data_shape == (height, width)

    roi = scanner.read().astype('float32') - 0x8000
    assert np.allclose(full[top:(top + height), left:(left + width)], roi)
