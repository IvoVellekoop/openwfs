from slm import SLM, enumerate_monitors
from patch import Patch
import numpy as np
from math import pi
import geometry
import textures

# construct a new SLM object and add a patch to it
numerical_aperture = 0.8
s1 = SLM(0, left=0, width=200, height=300)
s2 = SLM(0, left=500)
g = geometry.square(numerical_aperture)
g[1, 1, 1] = 0
p1 = Patch(s1, g)
pf = Patch(s1, geometry.square(1.0))
p2 = Patch(s2, geometry.square(1.0))
p3 = Patch(s2, geometry.square(0.2))
p4 = Patch(s2, geometry.square(0.1))
p3.phases = 0.25
p4.phases = 1
p4.additive_blend = False

pf.phases = textures.defocus(100) * 6
p1.enabled = False
rng = np.random.default_rng()
for n in range(50):
    data = rng.random([10, 10], np.float32) * 2.0 * pi
    p1.phases = data
    p2.phases = n/4.0
    s1.update()
    s2.update()

p1 = None  # test deletion
s1.patches.clear()
enumerate_monitors()
