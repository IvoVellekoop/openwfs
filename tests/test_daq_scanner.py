import numpy as np

from ..openwfs.devices import ScanningMicroscope
import time
import astropy.units as u
from astropy.units import Quantity
import matplotlib.pyplot as plt


def test_scanpattern_delay():
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz, axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev4/ai0', -1 * u.V, 1.0 * u.V),
                                 data_shape=(5, 5), scale=440 * u.um / u.V)
    pattern = scanner._scan_pattern
    print(pattern)


def test_daq_connection():
    sample_rate = 0.1 * u.MHz
    scanner = ScanningMicroscope(bidirectional=True, sample_rate=sample_rate, axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                 axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V), input=('Dev4/ai0', 0.1 * u.V, 1.0 * u.V),
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
    assert np.allclose(scanner.dwell_time, 1.0 / sample_rate)
    im1 = scanner.read()
    assert im1.dtype == np.dtype('uint16')
    assert im1.flags['C_CONTIGUOUS']
    im2 = scanner.read()
    scanner.width = 13
    scanner.height = 8
    scanner.left = 2
    scanner.top = 3
    assert scanner.width == 13
    assert scanner.height == 8
    assert scanner.left == 2
    assert scanner.top == 3
    time.sleep(1.09)
    im3 = scanner.read()
    scanner.width = 50
    scanner.height = 50
    plt.imshow(scanner.read())
    plt.colorbar()
    plt.show()


def test_pattern_reconstruction():
    padding_values = [0.2, 0.5]  # Example padding values
    delay_values = [0 * u.us, 10 * u.us]  # Example delay values
    data_shape = (10, 10)

    for padding_value in padding_values:
        for delay_value in delay_values:
            scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.1 * u.MHz,
                                         axis0=('Dev4/ao0', -1.0 * u.V, 1.0 * u.V),
                                         axis1=('Dev4/ao1', -1.0 * u.V, 1.0 * u.V),
                                         input=('Dev4/ai0', -1 * u.V, 1.0 * u.V),
                                         data_shape=data_shape, scale=440 * u.um / u.V,
                                         delay=delay_value, padding=padding_value)
            scanner._update()  # Generate the scan pattern

            # Simulate the transformation of the pattern into integers
            pattern_as_int = (scanner._scan_pattern[1] * 0x7FFF).astype(np.int16)

            # Process the pattern with _raw_to_cropped
            cropped_image = scanner._raw_to_cropped(pattern_as_int)

            # Verify dimensions of the cropped image
            assert cropped_image.shape == scanner._data_shape, \
                f"Cropped image dimensions do not match expected shape with padding {padding_value} and delay {delay_value}"

    # show that images are reconstructed correctly by using _raw_to_cropped on an integer version of the scan pattern
    plt.imshow(cropped_image)
    plt.figure()
    plt.imshow(scanner._raw_to_cropped((scanner._scan_pattern[0] * 0x7FFF).astype(np.int16)))
    plt.show()
