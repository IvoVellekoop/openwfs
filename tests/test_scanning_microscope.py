import astropy.units as u
import nidaqmx
import numpy as np
import pytest

from openwfs.devices import is_loaded

if not is_loaded(nidaqmx):
    pytest.skip(nidaqmx.messagea, allow_module_level=True)

from openwfs.devices import ScanningMicroscope, Axis
from openwfs.devices.galvo_scanner import InputChannel
from openwfs.utilities import coordinate_range


@pytest.mark.parametrize("start, stop", [(0.0, 1.0), (1.0, 0.0)])
def test_scan_axis(start, stop):
    """Tests if the Axis class generates the correct voltage sequences for stepping and scanning."""
    maximum_acceleration = 1 * u.V / u.ms**2
    scale = 440 * u.um / u.V
    v_min = -1.0 * u.V
    v_max = 2.0 * u.V
    a = Axis(
        channel="Dev4/ao0",
        v_min=v_min,
        v_max=v_max,
        maximum_acceleration=maximum_acceleration,
        scale=scale,
    )
    assert a.channel == "Dev4/ao0"
    assert a.v_min == v_min
    assert a.v_max == v_max
    assert a.maximum_acceleration == maximum_acceleration
    assert a.to_volt(0.0) == v_min
    assert a.to_volt(1.0) == v_max
    assert a.to_volt(2.0) == v_max  # test clipping
    assert a.to_volt(-0.1) == v_min  # test clipping

    # test step
    sample_rate = 0.5 * u.MHz
    step = a.step(start, stop, sample_rate)
    # plt.plot(step)
    # plt.show()
    assert np.isclose(step[0], -1.0 * u.V if start == 0.0 else 2.0 * u.V)
    assert np.isclose(step[-1], 2.0 * u.V if start == 0.0 else -1.0 * u.V)
    assert np.all(step >= v_min)
    assert np.all(step <= v_max)
    acceleration = np.diff(np.diff(step)) * sample_rate**2
    assert np.all(np.abs(acceleration) <= maximum_acceleration * 1.01)
    center = 0.5 * (start + stop)
    amplitude = 0.5 * (stop - start)

    # test clipping
    assert np.allclose(step, a.step(center - 1.1 * amplitude, center + 1.1 * amplitude, sample_rate))

    # test scan. Note that we cannot use the full scan range because we need
    # some time to accelerate / decelerate
    sample_count = 10000
    scan, launch, land, linear_region = a.scan(
        center - 0.8 * amplitude, center + 0.8 * amplitude, sample_count, sample_rate
    )
    half_pixel = 0.8 * amplitude / sample_count
    # plt.plot(scan)
    # plt.show()
    assert linear_region.stop - linear_region.start == sample_count
    assert linear_region.start == len(scan) - linear_region.stop
    assert np.isclose(scan[0], a.to_volt(launch))
    assert np.isclose(scan[-1], a.to_volt(land))
    assert np.isclose(scan[linear_region.start], a.to_volt(center - 0.8 * amplitude + half_pixel))
    assert np.isclose(scan[linear_region.stop - 1], a.to_volt(center + 0.8 * amplitude - half_pixel))
    speed = np.diff(scan[linear_region])
    assert np.allclose(speed, speed[0])  # speed should be constant

    acceleration = np.diff(np.diff(scan)) * sample_rate**2
    assert np.all(np.abs(acceleration) <= maximum_acceleration * 1.01)


def make_scanner(bidirectional, direction, reference_zoom):
    scale = 440 * u.um / u.V
    sample_rate = 0.5 * u.MHz
    input_channel = InputChannel(channel="Dev4/ai0", v_min=-1.0 * u.V, v_max=1.0 * u.V)
    y_axis = Axis(
        channel="Dev4/ao0",
        v_min=-2.0 * u.V,
        v_max=2.0 * u.V,
        maximum_acceleration=10 * u.V / u.ms**2,
        scale=scale,
    )
    x_axis = Axis(
        channel="Dev4/ao1",
        v_min=-2.0 * u.V,
        v_max=2.0 * u.V,
        maximum_acceleration=10 * u.V / u.ms**2,
        scale=scale,
    )
    return ScanningMicroscope(
        bidirectional=bidirectional,
        sample_rate=sample_rate,
        resolution=1024,
        input=input_channel,
        y_axis=y_axis,
        x_axis=x_axis,
        test_pattern=direction,
        reference_zoom=reference_zoom,
    )


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_scan_pattern(direction, bidirectional):
    """A unit test for scanning patterns."""
    reference_zoom = 1.2
    scanner = make_scanner(bidirectional, direction, reference_zoom)
    assert np.allclose(scanner.extent, scanner._x_axis.scale * 4.0 * u.V / reference_zoom)
    # plt.imshow(scanner.read())
    # plt.show()

    # check if returned pattern is correct
    y, x = coordinate_range(
        (scanner._resolution, scanner._resolution),
        10000 / reference_zoom,
        offset=(5000, 5000),
    )
    full = scanner.read().astype("float32") - 0x8000

    pixel_size = full[1, 1] - full[0, 0]

    if direction == "horizontal":
        assert np.allclose(full, full[0, :])  # all rows should be the same
        assert np.allclose(x, full, atol=0.2 * pixel_size)  # some rounding due to quantization
    else:
        # all columns should be the same (note we need to keep the last dimension for correct broadcasting)
        assert np.allclose(full, full[:, 0:1])
        assert np.allclose(y, full, atol=0.2 * pixel_size)

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

    roi = scanner.read().astype("float32") - 0x8000
    assert np.allclose(full[top : (top + height), left : (left + width)], roi, atol=0.2 * pixel_size)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_park_beam(bidirectional):
    """A unit test for parking the beam of a DAQ scanner."""
    reference_zoom = 1.2
    scanner = make_scanner(bidirectional, "horizontal", reference_zoom)

    # Park beam horizontally
    scanner.top = 3
    scanner.left = 4
    scanner.width = 1
    scanner.height = 2

    img = scanner.read()
    assert img.shape == (2, 1)
    voltages = scanner._scan_pattern
    assert np.allclose(voltages[1, :], voltages[1, 0])  # all voltages should be the same

    # Park beam vertically
    scanner.width = 2
    scanner.height = 1
    img = scanner.read()
    assert img.shape == (1, 2)
    voltages = scanner._scan_pattern
    assert np.allclose(voltages[0, :], voltages[0, 0])

    # Park beam horizontally and vertically
    scanner.width = 1
    scanner.height = 1
    img = scanner.read()
    assert img.shape == (1, 1)
    voltages = scanner._scan_pattern
    assert np.allclose(voltages[1, :], voltages[1, 0])  # all voltages should be the same
    assert np.allclose(voltages[0, :], voltages[0, 0])  # all voltages should be the same

    # TODO: add test for zooming
