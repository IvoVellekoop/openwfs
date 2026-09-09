import pytest
import numpy as np
from astropy import units as u
from openwfs.devices import SLMBlinkHDMI
import os.path

blink_path = r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\Blink_C_wrapper.dll"
os.path.isfile(blink_path) 
if not os.path.isfile(blink_path):
    pytest.skip("Blink SDK not found. Skipping tests.", allow_module_level=True)

num_slm = SLMBlinkHDMI.num_devices(blink_path)
if num_slm < 1:
    pytest.skip("No Meadowlark blink SLMs are connected. Skipping tests.", allow_module_level=True)

@pytest.fixture(scope="module")
def slm():
    """Fixture to create SLMBlinkHDMI instance for testing."""
    # Does not load the hardware_lookup_table. This is because otherwise we would not have a way to put the device as it was.
    slm_instance = SLMBlinkHDMI(
        blink_path=blink_path,
        monitor_id=2,
        coordinate_system="full",
        load_hardware_lookup_table=False,
        hardware_lookup_table=np.arange(1024),
        hidden=False
    )
    yield slm_instance
    del slm_instance


def test_slm_lookup_table_set(slm):
    """Test setting the lookup table."""
    lut = np.arange(2**slm.bit_depth)
    slm.lookup_table = lut
    np.testing.assert_array_equal(slm.lookup_table, lut)


def test_slm_set_phases(slm):
    """Test setting phases on the SLM."""
    slm.set_phases(np.pi)
    phi = slm.pixels.read()
    assert phi.shape == slm.shape
    np.testing.assert_allclose(phi, round(2**slm.bit_depth / 2), atol=0.5)


def test_slm_temperature(slm):
    """Test that SLM temperature is above 5°C."""
    assert 5 * u.deg_C < slm.temperature
