import time

from ..openwfs.simulation.mockdevices import MockSource, Generator
from ..openwfs.core import get_pixel_size
import numpy as np
import astropy.units as u


def test_mock_detector():
    image = np.ones((4, 5))
    source = MockSource(image, pixel_size=4 * u.um)
    data = source.read()
    assert np.allclose(image, data)
    assert get_pixel_size(data) == 4 * u.um


def test_timing_detector():
    image0 = np.zeros((4, 5))
    image1 = np.ones((4, 5))
    source = MockSource(image0, pixel_size=4 * u.um, measurement_duration=0.5 * u.s)
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

    assert np.allclose(t1 - t0, 0.0E9, atol=0.1E9)
    assert np.allclose(t2 - t1, 0.5E9, atol=0.1E9)
    assert np.allclose(t3 - t2, 0.0E9, atol=0.1E9)
    assert np.allclose(t4 - t3, 0.5E9, atol=0.1E9)
    assert np.allclose(t5 - t4, 0.0E9, atol=0.1E9)


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
