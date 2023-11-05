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
