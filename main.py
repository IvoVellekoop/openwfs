from slm import SLM, enumerate_monitors
from patch import Patch
import numpy as np
from math import pi
import geometry

# construct a new SLM object and add a patch to it
numerical_aperture = 0.8
s = SLM(0)
p = Patch(s, geometry.square(numerical_aperture))
#p = Patch(s, geometry.rectangle(0.1, 0.1, 0.2, 0.2))

rng = np.random.default_rng()
for n in range(10):
    data = rng.random([10, 10], np.float32) * 2.0 * pi
    s.patches[0].phases = data
    s.update()

enumerate_monitors()