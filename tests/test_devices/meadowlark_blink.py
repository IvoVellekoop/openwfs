import pytest
import numpy as np
from astropy import units as u
from openwfs.devices import SLMBlinkHDMI


@pytest.fixture
def slm():
    """Fixture to create SLMBlinkHDMI instance for testing."""
    slm_instance = SLMBlinkHDMI(
        blink_path=r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\Blink_C_wrapper.dll",
        monitor_id=2,
        coordinate_system="full",
        load_lookup_table=False,
        hardware_lookup_table=np.arange(1024),
    )
    yield slm_instance
    del slm_instance


def test_slm_lookup_table_set(slm):
    """Test setting the lookup table."""
    slm.lookup_table = np.arange(128)
    np.testing.assert_array_equal(slm.lookup_table, np.arange(128))


def test_slm_set_phases(slm):
    """Test setting phases on the SLM."""
    slm.set_phases(2 * np.pi - 0.005)
    phi = slm.pixels.read()
    assert phi.shape == (slm.height, slm.width)



def test_slm_temperature(slm):
    """Test that SLM temperature is above 5°C."""
    assert 5 * u.deg_C < slm.temperature
