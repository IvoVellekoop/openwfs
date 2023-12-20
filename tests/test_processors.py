import matplotlib.pyplot as plt
import pytest
import numpy as np
from ..openwfs.simulation.mockdevices import MockSource
from ..openwfs.processors import SingleRoi, SelectRoi, SelectRoiCircle
import skimage as sk
import astropy.units as u
from ..openwfs.utilities import imshow

def create_mock_source_with_data():
    # Create a mock source with predefined data for testing
    data = np.arange(100).reshape(10, 10)
    pixel_size = 1 * np.ones(2)  # Assuming 2D data with uniform pixel size
    return MockSource(data, pixel_size=pixel_size)

def test_croppers():
    img = sk.data.camera()
    src = MockSource(img, 50 * u.nm)
    roi = SelectRoi(src)
    roi2 = SelectRoiCircle(source=src)

def test_single_roi_simple_case():
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    pixel_size = 1 * np.ones(2)
    mock_source = MockSource(data, pixel_size=pixel_size)
    roi_processor = SingleRoi(mock_source, x=1, y=1, radius=1)
    roi_processor.trigger()
    result = roi_processor.read()

    # Debugging: Print the mask and the selected region
    print("Mask:", roi_processor._mask)

    expected_value = np.mean(data[0:3, 0:3])  # Assuming this is how the ROI is defined
    assert np.isclose(result, expected_value), f"ROI average value is incorrect. Expected: {expected_value}, Got: {result}"