import numpy as np

from ..openwfs.devices import ScanningMicroscope
import time
import astropy.units as u
from astropy.units import Quantity
import matplotlib.pyplot as plt


def test_scanpattern_delay():
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev1/ai0', -1 * u.V, 1.0 * u.V),
                                 data_shape=(5, 5), scale=440 * u.um / u.V)
    pattern = scanner._scan_pattern
    print(pattern)


def test_daq_connection():
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.1 * u.MHz, axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev1/ai0', -1 * u.V, 1.0 * u.V),
                                 data_shape=(15, 12), scale=440 * u.um / u.V, delay=9.0 * u.us, padding=0.1)

    assert scanner.top == 0
    assert scanner.left == 0
    assert scanner.height == 15
    assert scanner.width == 12
    assert scanner.data_shape == (15, 12)
    assert scanner.padding == 0.1
    ps0 = 440 * u.um / u.V * 2.0 * u.V / scanner.data_shape[0]
    ps1 = 440 * u.um / u.V * 2.0 * u.V / scanner.data_shape[1] * (1.0 - scanner.padding)
    assert np.allclose(scanner.pixel_size, (ps0, ps1))
    plt.imshow(scanner.read())
    plt.colorbar()
    plt.show()
