import logging
import time
import pytest
from ..openwfs.simulation.mockdevices import StaticSource, NoiseSource, SLM
from ..openwfs.utilities import set_pixel_size, get_pixel_size
from ..openwfs.processors import CropProcessor
import numpy as np
import astropy.units as u


def test_set_pixel_size():
    # Test case 1: Broadcast pixel size
    data = np.array([[1, 2], [3, 4]])
    pixel_size = 0.1 * u.m
    modified_data = set_pixel_size(data, pixel_size)
    assert np.all(get_pixel_size(modified_data) == (0.1, 0.1) * u.m)

    # Test case 2: Anisotropic pixel size
    data = np.array([[1, 2], [3, 4]])
    pixel_size = [0.1, 0.2] * u.m
    modified_data = set_pixel_size(data, pixel_size)
    assert np.all(get_pixel_size(modified_data) == pixel_size)

    # Test case 3: None pixel size
    data = np.array([[1, 2], [3, 4]])
    pixel_size = None
    modified_data = set_pixel_size(data, pixel_size)
    assert np.all(get_pixel_size(modified_data) == pixel_size)

    # Test case 4: Getting pixel size from bare numpy array
    data = np.array([[1, 2], [3, 4]])
    assert get_pixel_size(data) is None


@pytest.mark.parametrize("pixel_size", [4 * u.um, (4, 3) * u.um])
def test_mock_detector(pixel_size):
    image = np.ones((4, 5))
    source = StaticSource(image, pixel_size=pixel_size)
    data = source.read()
    data2 = source.trigger().result()
    data3 = np.empty(data.shape)
    source.trigger(out=data3)
    source.wait()
    assert np.allclose(image, data)
    assert np.allclose(image, data2)
    assert np.allclose(image, data3)
    ps = get_pixel_size(data)
    assert len(ps) == 2
    assert len(source.pixel_size)
    assert np.all(ps == source.pixel_size)
    assert np.all(ps == pixel_size)
    y = source.coordinates(0)
    x = source.coordinates(1)
    assert np.allclose(y, ([0.5, 1.5, 2.5, 3.5] * ps[0]).reshape((4, 1)))
    assert np.allclose(x, ([0.5, 1.5, 2.5, 3.5, 4.5] * ps[1]).reshape((1, 5)))
    assert np.allclose(source.extent, (4, 5) * ps)


@pytest.mark.parametrize("duration", [0.0 * u.s, 0.5 * u.s])
def test_timing_detector(caplog, duration):
    # note: this test will fail if multithreading is disabled
    caplog.set_level(logging.DEBUG)
    image0 = np.zeros((4, 5))
    image1 = np.ones((4, 5))
    source = StaticSource(image0, pixel_size=4 * u.um, duration=duration)
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
    source = NoiseSource('uniform', data_shape=(10, 11, 20), low=-1.0, high=1.0, pixel_size=4 * u.um)
    data = source.read()
    assert data.shape == (10, 11, 20)
    assert np.min(data) >= -1.0
    assert np.max(data) < 1.0
    assert np.allclose(np.mean(data), 0.0, atol=0.1)
    assert np.allclose(np.std(data), 2.0 / np.sqrt(12.0), atol=0.1)
    assert np.all(get_pixel_size(data) == 4 * u.um)
    source.data_shape = (2, 3)
    assert source.read().shape == (2, 3)


def test_mock_slm():
    slm = SLM((4, 4))
    slm.set_phases(0.5)
    assert np.allclose(slm.pixels.read(), round(0.5 * 256 / (2 * np.pi)), atol=0.5 / 256)
    discretized_phase = slm.phases.read()
    assert np.allclose(discretized_phase, 0.5, atol=1.1 * np.pi / 256)
    assert np.allclose(slm.field.read(), np.exp(1j * discretized_phase[0, 0]), rtol=2 / 256)
    slm.set_phases(np.array(((0.1, 0.2), (0.3, 0.4))), update=False)
    assert np.allclose(slm.phases.read(), 0.5, atol=1.1 * np.pi / 256)  # slm.update() not yet called, so should be 0.5
    slm.update()
    assert np.allclose(slm.phases.read(), np.array((
        (0.1, 0.1, 0.2, 0.2),
        (0.1, 0.1, 0.2, 0.2),
        (0.3, 0.3, 0.4, 0.4),
        (0.3, 0.3, 0.4, 0.4))), atol=1.1 * np.pi / 256)


def test_crop():
    data = np.random.uniform(size=(4, 5))
    source = StaticSource(data, pixel_size=1 * u.um)
    cropped = CropProcessor(source, padding_value=np.nan)
    assert cropped.data_shape == (4, 5)
    c = cropped.read()
    assert np.all(c == data)

    cropped.pos = (1, 2)
    assert cropped.data_shape == (4, 5)
    c2 = cropped.read()
    assert c2.shape == cropped.data_shape
    assert np.all(c2[0:-1, 0:-2] == data[1:, 2:])
    assert np.all(np.isnan(c2[-1, :]))
    assert np.all(np.isnan(c2[:, -2:]))

    cropped.pos = (-1, -2)
    cropped.data_shape = (2, 4)
    c3 = cropped.read()
    assert c3.shape == (2, 4)
    assert np.all(np.isnan(c3[0, :]))
    assert np.all(np.isnan(c3[0:1, :]))
    assert np.all(c3[1:, 2:] == data[0:1, 0:2])


def test_crop_1d():
    data = np.random.uniform(size=(10,))
    source = StaticSource(data, pixel_size=1 * u.um)
    cropped = CropProcessor(source, padding_value=np.nan)
    assert cropped.data_shape == (10,)
    c = cropped.read()
    assert np.all(c == data)

    cropped.pos = 4
    assert cropped.data_shape == (10,)
    assert cropped.pos == (4,)
    c2 = cropped.read()
    assert c2.shape == cropped.data_shape
    assert np.all(c2[:-4] == data[4:])
    assert np.all(np.isnan(c2[-4:]))

    cropped.pos = 4
    cropped.data_shape = 2
    assert cropped.data_shape == (2,)
    c3 = cropped.read()
    assert c3.shape == cropped.data_shape
    assert np.all(c3 == data[4:6])

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
