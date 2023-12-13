import numpy as np

from ..openwfs.devices import LaserScanning
import time
import astropy.units as u
import matplotlib.pyplot as plt


def test_scanpattern_delay():
    scanner = LaserScanning(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev4/ao2', -1.0 * u.V, 1.0 * u.V),
                            axis1=('Dev4/ao3', -1.0 * u.V, 1.0 * u.V), input=('Dev4/ai0', -1 * u.V, 1.0 * u.V),
                            data_shape=(5, 5), scale=440 * u.um / u.V)
    pattern = scanner._generate_scan_pattern()
    print(pattern)


def test_daq_connection():
    scanner = LaserScanning(bidirectional=True, sample_rate=500000 / u.s, x_mirror_mapping='Dev4/ao2',
                            pixel_size=880 / 40 * u.um,
                            y_mirror_mapping='Dev4/ao3', input_mapping='Dev4/ai0', delay=20 * u.ms,
                            scan_padding=60 * u.ms,
                            data_shape=(30, 20), input_min=-1 * u.V, input_max=1 * u.V,
                            voltage_mapping=440 * u.um / u.V)

    plt.imshow(scanner.read())
    plt.show()
