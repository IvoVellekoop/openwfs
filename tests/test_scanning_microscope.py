import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..openwfs.devices import ScanningMicroscope, Axis
from ..openwfs.utilities import place, set_pixel_size, get_pixel_size, coordinate_range


def test_scan_axis():
    """Tests if the Axis class generates the correct voltage sequences for stepping and scanning."""
    maximum_acceleration = 1 * u.V / u.ms ** 2
    a = Axis(channel='Dev4/ao0', v_min=-1.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=maximum_acceleration)
    assert a.channel == 'Dev4/ao0'
    assert a.v_min == -1.0 * u.V
    assert a.v_max == 2.0 * u.V
    assert a.maximum_acceleration == maximum_acceleration
    assert a.to_volt(0.0) == -1.0 * u.V
    assert a.to_volt(1.0) == 2.0 * u.V
    assert a.to_volt(2.0) == 2.0 * u.V  # test clipping
    assert a.to_volt(-0.1) == -1.0 * u.V  # test clipping

    # test step
    sample_rate = 0.5 * u.MHz
    step = a.step(0.0, 1.0, sample_rate)
    # plt.plot(step)
    # plt.show()
    assert np.isclose(step[0], -1.0 * u.V)
    assert np.isclose(step[-1], 2.0 * u.V)
    assert np.all(step >= -1.0 * u.V)
    assert np.all(step <= 2.0 * u.V)
    acceleration = np.diff(np.diff(step)) * sample_rate ** 2
    assert np.all(np.abs(acceleration) <= maximum_acceleration * 1.01)

    # test clipping
    assert np.allclose(step, a.step(-0.1, 2.1, sample_rate))

    # test scan. Note that we cannot use the full scan range because we need
    # some time to accelerate / decelerate
    sample_count = 10000
    scan, launch, land, linear_region = a.scan(0.2, 0.8, sample_count, sample_rate)
    plt.plot(scan)
    plt.show()
    assert linear_region.stop - linear_region.start == sample_count
    assert np.isclose(scan[0], a.to_volt(launch))
    assert np.isclose(scan[-1], a.to_volt(land))
    speed = np.diff(scan[linear_region])
    assert np.allclose(speed, speed[0])  # speed should be constant

    acceleration = np.diff(np.diff(scan)) * sample_rate ** 2
    assert np.all(np.abs(acceleration) <= maximum_acceleration * 1.01)


@pytest.mark.parametrize("direction", ['horizontal', 'vertical'])
def test_scan_pattern(direction):
    scale = 440 * u.um / u.V
    y_axis = Axis(channel='Dev4/ao0', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=1 * u.V / u.ms ** 2)
    x_axis = Axis(channel='Dev4/ao1', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=1 * u.V / u.ms ** 2)
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz,
                                 input=('Dev4/ai0', -1.0 * u.V, 1.0 * u.V), axis0=y_axis, axis1=x_axis,
                                 scale=scale, simulation=direction)

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
    assert np.allclose(zoomed, scaled - step / 2, atol=0.5 * step)

    scanner.zoom = 1.0
    reset_zoom = scanner.read().astype('float32') - 0x8000
    assert np.allclose(reset_zoom, roi)

    # test setting dwell time
    original_duration = scanner.duration
    scanner.delay = 1.0
    scanner.dwell_time = scanner.dwell_time * 2.0
    assert scanner.duration == original_duration * 2.0
    assert scanner.delay == 0.5
