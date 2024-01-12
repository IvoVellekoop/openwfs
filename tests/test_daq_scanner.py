import numpy as np

from ..openwfs.devices import ScanningMicroscope
import astropy.units as u
import matplotlib.pyplot as plt
import nidaqmx


def test_scan_pattern_delay():
    try:
        scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz,
                                     axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                     axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev4/ai0', -1 * u.V, 1.0 * u.V),
                                     data_shape=(5, 5), scale=440 * u.um / u.V)
        pattern = scanner._scan_pattern
        print(pattern)
    except (nidaqmx.DaqError, FileNotFoundError):
        print('No nidaq card found or NI-DAQ MAX not installed')
        pass


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
    except (nidaqmx.DaqError, FileNotFoundError):
        pass
