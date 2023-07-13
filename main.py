from slm import SLM, Patch, geometry, textures
import numpy as np
from math import pi

# construct a new SLM object and add a patch to it
numerical_aperture = 0.8
s1 = SLM(0, left=0, width=200, height=300)
s2 = SLM(0, left=300)
g = geometry.square(numerical_aperture)
g[1, 1, 1] = 0
#p1 = Patch(s1, g)
s1.primary_phase_patch.geometry = g
pf = Patch(s1)
p2 = Patch(s2)
p3 = Patch(s2, geometry.square(0.2))
p4 = Patch(s2, geometry.square(0.1))
p3.phases = 0.25
p4.phases = 1
p4.additive_blend = False

pf.phases = textures.defocus(100) * 6
rng = np.random.default_rng()
for n in range(50):
    data = rng.random([10, 10], np.float32) * 2.0 * pi
    s1.phases = data
    s2.phases[0] = n/4.0
    s2.phases = s2.phases  # manual synchronization still needed
    s1.update()
    s2.update()

p1 = None  # test deletion. After deleting the two windowed slms, we can create a new full screen one
s1.patches.clear()
s1 = 0
s2 = 0
s3 = SLM(1)