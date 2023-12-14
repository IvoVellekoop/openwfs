import numpy as np

from ..openwfs.devices import ScanningMicroscope
import time
import astropy.units as u
import matplotlib.pyplot as plt


def test_scanpattern_delay():
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev1/ai0', -1 * u.V, 1.0 * u.V),
                                 data_shape=(5, 5), scale=440 * u.um / u.V)
    pattern = scanner._scan_pattern
    print(pattern)


def test_daq_connection():
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev1/ai0', -1 * u.V, 1.0 * u.V),
                                 data_shape=(15, 12), scale=440 * u.um / u.V, delay=0.0 * u.us)

    plt.imshow(scanner.read())
    plt.colorbar()
    plt.show()
