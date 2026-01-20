import openwfs.devices as ow_d
import astropy.units as u
import numpy as np
import time

stage = ow_d.KCubeInertial()

for i in [True, False]:
    stage.pair_channels = i

    acc = 20000/u.s**2 * np.ones(4)
    vel = 500/u.s * np.ones(4)
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
# stage = 3
# stage.shutdown()

#
# class Dummy:
#     def __init__(self):
#         self._workers = ThreadPoolExecutor(max_workers=1)
#         self.value = None
#
#     def compute_value(self, *args_, **kwargs_):
#         print(self.value)
#         assert 1== 2
#         return args_[0] * args_[0]
#
#     def set_value(self, v):
#         self._future = self._workers.submit(self.compute_value, v, 2 * v)
#
#     def get_value(self):
#         print(self._future.done())
#         return self._future.result()
#
# d = Dummy()
# d.set_value(5)
# d._future.done()
# print(d.get_value\:0
