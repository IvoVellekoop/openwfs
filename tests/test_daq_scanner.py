import numpy as np
import pytest

openwfs_devices = pytest.importorskip("openwfs.devices")
from openwfs.devices import LaserScanning
import time
import astropy.units as u
import matplotlib.pyplot as plt

def test_scanpattern_padding():
    scanner = LaserScanning(bidirectional=False, scan_padding=2, data_shape=(5, 5), input_min=-2, input_max=2)
    pattern = scanner.scanpattern()
    x_coordinates = pattern[0, :-1].reshape((5, 7))  # 5 steps + 2 padding
    # First two elements of each row should be the same due to padding
    for row in x_coordinates:
        assert row[0] == row[1]

def test_scanpattern_bidirectional_flip():
    scanner = LaserScanning(bidirectional=True, scan_padding=0, data_shape=(5, 5), input_min=-2, input_max=2)
    pattern = scanner.scanpattern()
    x_coordinates = pattern[0, :-1].reshape((5, 5))

    # Check if odd rows are flipped
    for i in range(1, 5, 2):
        assert np.array_equal(x_coordinates[i], np.flip(x_coordinates[i - 1]))

def test_pmt_to_image_no_padding():
    scanner = LaserScanning(scan_padding=0, bidirectional=False, data_shape=(5, 5), input_min=-2, input_max=2)
    test_data = np.arange(26)  # Create test data
    image = scanner.pmt_to_image(test_data)

    # Ensure the data is reshaped correctly to a 5x5 image
    assert image.shape == (5, 5)
    assert np.array_equal(image, test_data[1:26].reshape((5, 5)))

def test_pmt_to_image_with_padding():
    padding = 2
    delay = 0
    scanner = LaserScanning(scan_padding=padding, delay=delay, bidirectional=True, data_shape=(5, 5), input_min=-2, input_max=2)
    test_data = np.arange(26 + 5*padding)  # Create test data with padding and extra point
    image = scanner.pmt_to_image(test_data)

    # Check that the shape accounts for padding
    assert image.shape == (5, 5)
    # Check that the padding has been accounted for by testing the first row
    assert np.array_equal(image[0], test_data[1:6]+padding)

def test_pmt_to_image_with_delay():
    padding = 0
    delay = 2
    scanner = LaserScanning(scan_padding=padding, delay=delay, bidirectional=True, data_shape=(5, 5), input_min=-2, input_max=2)
    test_data = np.arange(26 + 5*padding)  # Create test data with padding and extra point
    image = scanner.pmt_to_image(test_data)
    # Check that the shape accounts for padding
    assert image.shape == (5, 5)

    # Check that the delay has been accounted for by testing the first row
    check_array = test_data[1:6]-delay
    check_array[check_array<0] = 0
    assert np.array_equal(image[0], check_array)

def test_scanpattern_and_pmt_to_image_with_changed_dimensions():
    # Change width and height
    new_width, new_height = 10, 16  # New dimensions
    scanner = LaserScanning(scan_padding=0, bidirectional=True, data_shape=(new_height, new_width), input_min=-2, input_max=2)

    # Test scanpattern
    pattern = scanner.scanpattern()
    x_coordinates = pattern[0, :-1].reshape((new_height, new_width))

    for i in range(1, new_height, 2):
        assert np.array_equal(x_coordinates[i], np.flip(x_coordinates[i - 1]))

    # Test pmt_to_image
    test_data = np.arange(new_width * new_height + 1)  # Adjust test data for new dimensions

    image = scanner.pmt_to_image(test_data)
    assert image.shape == (new_height, new_width)
    test_image = test_data[1:].reshape((new_height, new_width))
    test_image[1::2, :] = test_image[1::2, ::-1]
    assert np.array_equal(image, test_image)

def test_scanpattern_and_pmt_to_image_with_padding_and_changed_dimensions():
    # Change width, height, and add padding
    new_width, new_height, padding = 10, 16, 2  # New dimensions and padding
    scanner = LaserScanning(scan_padding=padding, bidirectional=True, data_shape=(new_height, new_width), input_min=-2, input_max=2)

    # Test scanpattern
    pattern = scanner.scanpattern()
    padded_width = new_width + padding
    x_coordinates = pattern[0, :-1].reshape((new_height, padded_width))
    plt.imshow(x_coordinates)
    plt.show()
    for i in range(new_height):
        if i % 2 == 0:
            # Even rows: the first 'padding' values should be the same (start padding)
            assert np.all(x_coordinates[i, :padding] == x_coordinates[i, 0])
        else:
            # Odd rows: the last 'padding' values should be the same (end padding) and check flipping
            assert np.all(x_coordinates[i, -padding:] == x_coordinates[i, -1])
            assert np.array_equal(x_coordinates[i, padding:], np.flip(x_coordinates[i - 1, :new_width]))

    # Test pmt_to_image
    test_data_length = new_width * new_height + padding * new_height + 1  # Adjust test data for new dimensions and padding
    test_data = np.arange(test_data_length)

    image = scanner.pmt_to_image(test_data)
    plt.imshow(image)
    plt.show()
    assert image.shape == (new_height, new_width)

    # Prepare the expected image for comparison
    test_image = np.zeros((new_height, new_width))
    for i in range(new_height):
        if i % 2 == 0:
            test_image[i, :] = test_data[1 + i * padded_width: 1 + i * padded_width + new_width]
        else:
            test_image[i, :] = test_data[1 + i * padded_width + padding: 1 + i * padded_width + padding + new_width][::-1]

    assert np.array_equal(image, test_image)


def test_measurement_time():
    # This tests if the measurement time actually changes the time the scanners measures. Requires connection to DAQ
    # or a simulated DAQ in NI MAX

    scanner = LaserScanning(duration=10 * u.ms, x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3',
                            input_mapping='Dev4/ai24')
    diffs = []
    times = []
    for timers in range(1,10):

        scanner.duration = timers/10 * u.s

        t = time.time()
        scanner.read()

        diffs.append((time.time() - t))
        times.append(timers/10 * u.s)


    diffs = diffs

    for diff,ts in zip(diffs,times):
        assert diff-ts.value < 0.2