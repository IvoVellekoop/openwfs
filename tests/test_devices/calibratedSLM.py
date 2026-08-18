from openwfs.simulation.calibratedSLM import CalibratedSLM
import astropy.units as u
import numpy as np


def test_calibrated_slm_initialization():
    """
    Test the initialization of the CalibratedSLM class.
    """
    physical_size = (2 * u.um, 2 * u.um)
    shape = (100, 100)

    cslm = CalibratedSLM(physical_size=physical_size, shape=shape, monitor_id=0)

    assert cslm.physical_size == physical_size
    assert cslm.shape == shape
    assert cslm.monitor_id == 0

    cslm.wavelength = 500 * u.nm
    assert cslm.wavelength == 500 * u.nm, "Wavelength setter did not update the wavelength correctly."

    cslm.physical_size = (3 * u.um, 3 * u.um)
    assert cslm.physical_size == (
        3 * u.um,
        3 * u.um,
    ), "Physical size setter did not update the physical size correctly."


def test_calibrated_slm_phases():
    """
    Test setting and reading phases in the CalibratedSLM class.
    """
    physical_size = (2 * u.um, 2 * u.um)
    shape = (100, 100)

    cslm = CalibratedSLM(physical_size=physical_size, shape=shape, monitor_id=0)

    # Set random phases
    rnd = np.random.RandomState(42)
    random_phases = rnd.rand(*shape) * 2 * np.pi
    cslm.set_phases(random_phases)

    # Read back the phases
    read_phases = cslm.phases.read()
    assert np.allclose(random_phases, read_phases), "The set and read phases do not match."

    # Read back the field
    read_field = cslm.field.read()
    expected_field = np.exp(1j * random_phases)
    assert np.allclose(expected_field, read_field), "The set and read fields do not match."


def test_cslm_field_amplitude():
    """
    Test the field amplitude of the CalibratedSLM class.
    """
    field_amplitude = np.ones((100, 100)) * 10
    field_amplitude[30:70, 30:70] = 0
    cslm = CalibratedSLM(
        physical_size=(2 * u.um, 2 * u.um),
        amplitude=field_amplitude,
        shape=(100, 100),
        monitor_id=0,
    )

    assert np.allclose(cslm.amplitude, field_amplitude), "The field amplitude does not match the expected value."

    # should throw an error if the amplitude shape does not match the SLM shape
    try:
        field_amplitude = np.ones((99, 99)) * 10
        field_amplitude[30:70, 30:70] = 0
        cslm = CalibratedSLM(
            physical_size=(2 * u.um, 2 * u.um),
            amplitude=field_amplitude,
            shape=(100, 100),
            monitor_id=0,
        )
    except ValueError as e:
        assert str(e) == "amplitude must have the same shape as the SLM shape."
