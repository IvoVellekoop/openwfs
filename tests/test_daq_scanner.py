import numpy as np
import pytest

from ..openwfs.devices import ScanningMicroscope
import astropy.units as u
import matplotlib.pyplot as plt
import nidaqmx

# older versions of nidaqmx report a FileNotFoundError instead of a DaqNotFoundError
DaqNotFoundError = nidaqmx.errors.DaqNotFoundError if (
        hasattr(nidaqmx, 'errors') and hasattr(nidaqmx.errors, 'DaqNotFoundError')) else FileNotFoundError


def test_scan_pattern_delay():
    try:
        scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz,
                                     axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                     axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V),
                                     input=('Dev4/ai0', -1.0 * u.V, 1.0 * u.V),
                                     data_shape=(5, 5), scale=440 * u.um / u.V)
        pattern = scanner._scan_pattern
        print(pattern)
    except (nidaqmx.DaqError, DaqNotFoundError):
        print('No NI-DAQ card found or NI-DAQ MAX not installed')
        pytest.skip()


def test_daq_connection():
    try:
        sample_rate = 0.1 * u.MHz
        scanner = ScanningMicroscope(bidirectional=True, sample_rate=sample_rate,
                                     axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                     axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V),
                                     input=('Dev4/ai0', 0.1 * u.V, 1.0 * u.V),
                                     data_shape=(15, 12), scale=440 * u.um / u.V, delay=9.0, padding=0.1)

        assert scanner.top == 0
        assert scanner.left == 0
        assert scanner.height == 15
        assert scanner.width == 12
        assert scanner.data_shape == (15, 12)
        assert scanner.padding == 0.1
        ps0 = 440 * u.um / u.V * 2.0 * u.V / scanner.data_shape[0]
        ps1 = 440 * u.um / u.V * 2.0 * u.V / scanner.data_shape[1] * (1.0 - scanner.padding)
        assert np.allclose(scanner.pixel_size, (ps0, ps1))
        assert np.allclose(scanner.dwell_time, 1.0 / sample_rate)
        im1 = scanner.read()
        assert im1.dtype == np.dtype('uint16')
        assert im1.flags['C_CONTIGUOUS']

        # read second image
        im2 = scanner.read()  # noqa
        scanner.width = 13
        scanner.height = 8
        scanner.left = 2
        scanner.top = 3
        assert scanner.width == 13
        assert scanner.height == 8
        assert scanner.left == 2
        assert scanner.top == 3

        im3 = scanner.read()  # noqa
        scanner.width = 50
        scanner.height = 50
        plt.imshow(scanner.read())
        plt.colorbar()
        plt.show()
    except (nidaqmx.DaqError, DaqNotFoundError):
        print('No NI-DAQ card found or NI-DAQ MAX not installed')
        pytest.skip()


def test_park_beam():
    """A unit test for parking the beam of a DAQ scanner."""
    try:
        sample_rate = 0.05 * u.MHz
        scanner = ScanningMicroscope(bidirectional=True, sample_rate=sample_rate,
                                     axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                     axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V),
                                     input=('Dev4/ai0', 0.1 * u.V, 1.0 * u.V),
                                     data_shape=(40, 30), scale=440 * u.um / u.V, delay=9.0, padding=0.1)
    except (nidaqmx.DaqError, DaqNotFoundError):
        print('No NI-DAQ card found or NI-DAQ MAX not installed')
        pytest.skip()
        return

    # Choose ROI of a single point
    scanner.top = 3
    scanner.left = 4
    scanner.width = 1
    scanner.height = 1

    img = scanner.read()
    assert img.size == 1
