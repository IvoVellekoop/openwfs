import openwfs.devices as ow_d
import astropy.units as u
import numpy as np
import time

# Test KCubeInertial stage. Requires a physical device KIM101 to be connected.

stage = ow_d.KCubeInertial()
print("asda")

for i in [True, False]:
    stage.pair_channels = i

    acc = 20000 / u.s**2 * np.ones(4)
    vel = 500 / u.s * np.ones(4)
    stage.acceleration = acc
    stage.velocity = vel

    assert np.allclose(stage.velocity, vel)
    assert np.allclose(stage.acceleration, acc)

    p_i = np.ones(4) * 10
    stage.position = np.zeros(4) * 10
    stage.stop()
    # stage.position = np.ones(4) * 10 # Should through an error
    stage.wait()
    stage.position = np.ones(4) * 10
    stage.wait()
    # print(stage.position)
    assert np.allclose(stage.position, p_i)

    p_f = np.array([10, 10, 10, 10]) * 10
    stage.position = p_f
    stage.wait()
    assert np.allclose(stage.position, p_f)

    delta = np.array([10, -10, 10, -10])
    stage.move_by(delta)
    stage.wait()
    assert np.allclose(stage.position, p_f + delta)

del stage
