import numpy as np
import astropy.units as u

from ..openwfs.simulation import SLM
from ..openwfs.simulation import Stage
from astropy.units import Quantity


def test_mock_slm():
    # TODO: at some point, merge this with the test_slm.py file
    # create a mock SLM object with a given shape
    slm = SLM(shape=(480, 640))

    # clear the slm
    slm.set_phases(0)

    # test if all elements of slm.phases are 0
    assert np.all(slm.phases.read() == 0)

    # set a pattern, don't update yet
    pattern = np.random.uniform(0.0, 1.9 * np.pi, size=(48, 64))
    slm.set_phases(pattern, update=False)

    # test if all elements of slm.phases are still 0
    assert np.all(slm.phases.read() == 0)

    # update the slm, and read back
    slm.update()
    scaled_pattern = np.repeat(np.repeat(pattern, 10, axis=0), 10, axis=1)
    assert np.allclose(slm.phases.read(), scaled_pattern, atol=1.0 * np.pi / 256)

def test_single_stage():
    stage = Stage("x", Quantity(10, u.um))
    assert stage.position == 0
    assert stage.axis == "x"

    stage.position = Quantity(5, u.um)
    assert stage.position == Quantity(0, u.um)

    stage.position = Quantity(15, u.um)
    assert stage.position == Quantity(20, u.um)

    stage.position = Quantity(-5, u.um)
    assert stage.position == Quantity(0, u.um)

    stage.position = Quantity(-15, u.um)
    assert stage.position == Quantity(-20, u.um)