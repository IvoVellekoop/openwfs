from ..openwfs.simulation import SimulatedWFS
import numpy as np
from ..openwfs.feedback import SingleRoiSquare, SelectRoiSquare, SelectRoiCircle
from skimage import data
import pytest


def test_square_selection_detector():
    """
    Test if the square selection detector is working as expected.
    """
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    width = 20
    height = 20
    detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
    detector.trigger()  #

    # Assert condition for detector functionality
    assert sim.read()[250, 250] / (width * height) == detector.read(), "Square detector not working as expected"


def square_selection_detector_test():
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat

    width = 20
    height = 20
    detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
    detector.trigger()  #
    if sim.read()[250, 250] / (width * height) != detector.read():
        raise Exception(f"Square detector not working as expected")
    return True


@pytest.fixture
def mock_simulated_wfs():
    sim = SimulatedWFS()
    sim.read = lambda: data.camera()  # Mock the read function to return a fixed image
    return sim


def test_select_roi_circle(mock_simulated_wfs):
    """
    Test the SelectRoiCircle detector functionality.
    """
    detector = SelectRoiCircle(mock_simulated_wfs)
    detector.trigger()

    circle_image = detector.read_circle()

    assert circle_image is not None
    assert circle_image.shape == mock_simulated_wfs.read().shape, "Circle ROI read image shape mismatch"


def test_select_roi_square(mock_simulated_wfs):
    """
    Test the SelectRoiSquare detector functionality.
    """
    detector = SelectRoiSquare(mock_simulated_wfs)
    detector.trigger()

    square_image = detector.read_square()

    # Check the properties of square_image
    assert square_image is not None
