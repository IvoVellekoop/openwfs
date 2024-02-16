import numpy as np

from ..openwfs.simulation import SLM


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
