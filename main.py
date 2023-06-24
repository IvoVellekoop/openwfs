from slm import SLM, enumerate_monitors
from patch import Patch
import numpy as np
from math import pi
import geometry

# construct a new SLM object and add a patch to it
numerical_aperture = 0.8
s1 = SLM(0, x=0)
s2 = SLM(0, x=500)
g = geometry.square(numerical_aperture)
g[1, 1, 1] = 0
p1 = Patch(s1, g)
p2 = Patch(s2, geometry.square(1.0))

rng = np.random.default_rng()
for n in range(20):
    data = rng.random([10, 10], np.float32) * 2.0 * pi
    p1.phases = data
    p2.phases = [[n]]
    s1.update()
    s2.update()

enumerate_monitors()
