import numpy as np

from ..openwfs.devices import ScanningMicroscope
from ..openwfs.slm.patterns import coordinate_range
from ..openwfs.utilities import imshow, place, set_pixel_size, get_pixel_size
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

    # test binning
    scanner.binning = 2
    assert scanner.binning == 2
    assert scanner.data_shape == (shape[0] / 2, shape[1] / 2)
    binned = scanner.read().astype('float32') - 0x8000
    assert np.allclose(0.5 * (full[::2, ::2] + full[1::2, 1::2]), binned)
    scanner.binning = 1
    restored = scanner.read().astype('float32') - 0x8000
    assert np.allclose(full, restored)

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

    # test zooming
    ps = scanner.pixel_size
    scanner.zoom = 2.0
    assert np.allclose(scanner.pixel_size, ps * 0.5)
    assert scanner.width == width
    assert scanner.height == height
    assert scanner.data_shape == (height, width)
    assert scanner.left == np.floor(2 * left + 0.5 * width)
    assert scanner.top == np.floor(2 * top + 0.5 * height)

    zoomed = scanner.read().astype('float32') - 0x8000
    scaled = place(zoomed.shape, 0.5 * ps, set_pixel_size(roi, ps))
    assert np.allclose(get_pixel_size(scaled), 0.5 * ps)
    step = zoomed[1, 1] - zoomed[0, 0]
    assert np.allclose(zoomed, scaled, atol=0.5 * step)

    scanner.zoom = 1.0
    reset_zoom = scanner.read().astype('float32') - 0x8000
    assert np.allclose(reset_zoom, roi)

    # test setting dwell time
    original_duration = scanner.duration
    scanner.delay = 1.0
    scanner.dwell_time = scanner.dwell_time * 2.0
    assert scanner.duration == original_duration * 2.0
    assert scanner.delay == 0.5
