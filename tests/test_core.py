import time
import pytest
from ..openwfs.simulation.mockdevices import MockSource, Generator, MockSLM
from ..openwfs.core import get_pixel_size
from ..openwfs.processors import CropProcessor
import numpy as np
import astropy.units as u


def test_mock_detector():
    image = np.ones((4, 5))
    source = MockSource(image, pixel_size=4 * u.um)
    data = source.read()
    data2 = source.trigger().result()
    data3 = np.empty(data.shape)
    source.trigger(out=data3)
    source.wait()
    assert np.allclose(image, data)
    assert np.allclose(image, data2)
    assert np.allclose(image, data3)
    assert get_pixel_size(data) == 4 * u.um


@pytest.mark.parametrize("duration", [0.0 * u.s, 0.5 * u.s])
def test_timing_detector(duration):
    image0 = np.zeros((4, 5))
    image1 = np.ones((4, 5))
    source = MockSource(image0, pixel_size=4 * u.um, duration=duration)
    t0 = time.time_ns()
    f0 = source.trigger()
    t1 = time.time_ns()
    source.data = image1  # waits for data acquisition to complete
    t2 = time.time_ns()
    f1 = source.trigger()
    t3 = time.time_ns()
    assert np.allclose(f1.result(), image1)
    t4 = time.time_ns()
    assert np.allclose(f0.result(), image0)
    t5 = time.time_ns()

    assert np.allclose(t1 - t0, 0.0, atol=0.1E9)
    assert np.allclose(t2 - t1, duration.to_value(u.ns), atol=0.1E9)
    assert np.allclose(t3 - t2, 0.0, atol=0.1E9)
    assert np.allclose(t4 - t3, duration.to_value(u.ns), atol=0.1E9)
    assert np.allclose(t5 - t4, 0.0, atol=0.1E9)


def test_noise_detector():
    source = Generator.uniform_noise(data_shape=(10, 11, 20), low=-1.0, high=1.0, pixel_size=4 * u.um)
    data = source.read()
    assert data.shape == (10, 11, 20)
    assert np.min(data) >= -1.0
    assert np.max(data) < 1.0
    assert np.allclose(np.mean(data), 0.0, atol=0.1)
    assert np.allclose(np.std(data), 2.0 / np.sqrt(12.0), atol=0.1)
    assert get_pixel_size(data) == 4 * u.um
    source.data_shape = (2, 3)
    assert source.read().shape == (2, 3)


def test_mock_slm():
    slm = MockSLM((4, 4))
    slm.set_phases(0.5)
    assert np.allclose(slm.pixels().read(), 0.5)
    slm.set_phases(np.array(((0.1, 0.2), (0.3, 0.4))), update=False)
    assert np.allclose(slm.pixels().read(), 0.5)
    slm.update()
    assert np.allclose(slm.pixels().read(), np.array((
        (0.1, 0.1, 0.2, 0.2),
        (0.1, 0.1, 0.2, 0.2),
        (0.3, 0.3, 0.4, 0.4),
        (0.3, 0.3, 0.4, 0.4))))


def test_crop():
    data = np.random.uniform(size=(4, 5))
    source = MockSource(data, pixel_size=1 * u.um)
    cropped = CropProcessor(source, padding_value=np.nan)
    assert cropped.data_shape == (4, 5)
    c = cropped.read()
    assert np.alltrue(c == data)

    cropped.pos = (1, 2)
    assert cropped.data_shape == (4, 5)
    c2 = cropped.read()
    assert c2.shape == cropped.data_shape
    assert np.alltrue(c2[0:-1, 0:-2] == data[1:, 2:])
    assert np.alltrue(np.isnan(c2[-1, :]))
    assert np.alltrue(np.isnan(c2[:, -2:]))

    cropped.pos = (-1, -2)
    cropped.data_shape = (2, 4)
    c3 = cropped.read()
    assert c3.shape == (2, 4)
    assert np.alltrue(np.isnan(c3[0, :]))
    assert np.alltrue(np.isnan(c3[0:1, :]))
    assert np.alltrue(c3[1:, 2:] == data[0:1, 0:2])


def test_crop_1D():
    data = np.random.uniform(size=(10,))
    source = MockSource(data, pixel_size=1 * u.um)
    cropped = CropProcessor(source, padding_value=np.nan)
    assert cropped.data_shape == (10,)
    c = cropped.read()
    assert np.alltrue(c == data)

    cropped.pos = 4
    assert cropped.data_shape == (10,)
    assert cropped.pos == (4,)
    c2 = cropped.read()
    assert c2.shape == cropped.data_shape
    assert np.alltrue(c2[:-4] == data[4:])
    assert np.alltrue(np.isnan(c2[-4:]))

    cropped.pos = 4
    cropped.data_shape = 2
    assert cropped.data_shape == (2,)
    c3 = cropped.read()
    assert c3.shape == cropped.data_shape
    assert np.alltrue(c3 == data[4:6])

# TODO: translate the tests below.
#  They should test the SingleROI processor, checking if the returned averaged value is correct.
#
# def test_square_selection_detector():
#     """
#     Test if the square selection detector is working as expected.
#     """
#     sim = SimulatedWFS()
#     sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
#
#     width = 20
#     height = 20
#     detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
#     detector.trigger()  #
#
#     # Assert condition for detector functionality
#     assert sim.read()[250, 250] / (width * height) == detector.read(), "Square detector not working as expected"
#
#
# def square_selection_detector_test():
#     sim = SimulatedWFS()
#     sim.set_ideal_wf(np.zeros([500, 500]))  # correct wf = flat
#
#     width = 20
#     height = 20
#     detector = SingleRoiSquare(sim, width=width, height=height, top=240, left=240)
#     detector.trigger()  #
#     if sim.read()[250, 250] / (width * height) != detector.read():
#         raise Exception(f"Square detector not working as expected")
#     return True
#
#
# @pytest.fixture
# def mock_simulated_wfs():
#     sim = SimulatedWFS()
#     sim.read = lambda: data.camera()  # Mock the read function to return a fixed image
#     return sim
#
#
# def test_select_roi_circle(mock_simulated_wfs):
#     """
#     Test the SelectRoiCircle detector functionality.
#     """
#     detector = SelectRoiCircle(mock_simulated_wfs)
#     detector.trigger()
#
#     circle_image = detector.read_circle()
#
#     assert circle_image is not None
#     assert circle_image.shape == mock_simulated_wfs.read().shape, "Circle ROI read image shape mismatch"
#
#
# def test_select_roi_square(mock_simulated_wfs):
#     """
#     Test the SelectRoiSquare detector functionality.
#     """
#     detector = SelectRoi(mock_simulated_wfs)
#     detector.trigger()
#
#     square_image = detector.read_square()
#
#     # Check the properties of square_image
#     assert square_image is not None