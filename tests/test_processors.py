import pytest
import numpy as np
from ..openwfs.simulation.mockdevices import MockSource
from ..openwfs.processors import SingleRoi, select_roi, Roi, MultipleRoi, MaskType
import skimage as sk
import astropy.units as u
from ..openwfs.utilities import imshow


def test_croppers():
    img = sk.data.camera()
    src = MockSource(img, 50 * u.nm)
    roi = select_roi(src, MaskType.DISK)
    assert roi.mask_type == MaskType.DISK


def test_single_roi_simple_case():
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    pixel_size = 1 * np.ones(2)
    mock_source = MockSource(data, pixel_size=pixel_size)
    roi_processor = SingleRoi(mock_source, radius=np.sqrt(2))
    roi_processor.trigger()
    result = roi_processor.read()

    # Debugging: Print the mask and the selected region
    print("Mask:", roi_processor._rois[()]._mask)

    expected_value = np.mean(data[0:3, 0:3])  # Assuming this is how the ROI is defined
    assert np.isclose(result,
                      expected_value), f"ROI average value is incorrect. Expected: {expected_value}, Got: {result}"


def create_mock_source_with_data():
    data = np.arange(25).reshape(5, 5)
    return MockSource(data, pixel_size=1 * u.um)


@pytest.mark.parametrize("x, y, radius, expected_avg", [
    (2, 2, 1, 12),  # Center ROI in 5x5 matrix
    (0, 0, 0, 0)  # Top-left corner ROI in 5x5 matrix
])
def test_single_roi(x, y, radius, expected_avg):
    mock_source = create_mock_source_with_data()
    roi_processor = SingleRoi(mock_source, (y, x), radius)
    roi_processor.trigger()
    result = roi_processor.read()

    assert np.isclose(result, expected_avg), f"ROI average value is incorrect. Expected: {expected_avg}, Got: {result}"


def test_multiple_roi_simple_case():
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    pixel_size = 1 * np.ones(2)
    mock_source = MockSource(data, pixel_size=pixel_size)

    rois = [Roi((1, 1), radius=0),
            Roi((2, 2), radius=0),
            Roi((1, 1), radius=1),
            Roi((0, 1), radius=0)
            ]

    roi_processor = MultipleRoi(mock_source, rois=rois)
    roi_processor.trigger()
    result = roi_processor.read()

    expected_values = [5, 9, 5, 2]
    assert all(np.isclose(r, e) for r, e in zip(result, expected_values)), \
        f"ROI average values are incorrect. Expected: {expected_values}, Got: {result}"
