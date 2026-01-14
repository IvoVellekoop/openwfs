import openwfs.devices as ow_d
import numpy as np

stage = ow_d.KCubeInertial(serial_number="97251304", channels = [2, 3], pair_channels=True)

p_i = np.zeros(2)
stage.position = np.zeros(2)
stage.wait()
assert np.allclose(stage.position, p_i)

p_f = np.array([10, 10])
stage.position = p_f
stage.wait()
assert np.allclose(stage.position, p_f)

delta = np.array([5, -3])
stage.move_by(delta)
stage.wait()
assert np.allclose(stage.position, p_f + delta)

del stage
# stage = 3
# stage.shutdown()
