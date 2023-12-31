import numpy as np
import time
import set_path  # noqa - needed for setting the module search path to find openwfs
from openwfs.slm import SLM, Patch, geometry, patterns
from skimage import data
import astropy.units as u

"""
Example on how different geometries and patches work for an SLM. Currently uses SLM number 0, which is the left
upper corner of the primary monitor.

EPILEPSY WARNING: YOUR PRIMARY SCREEN MAY QUICKLY FLASH DURING RUNNING THIS FILE
"""

# construct a new SLM object and add a patch to it
numerical_aperture = 0.8
s1 = SLM(0, shape=(300, 200))
s2 = SLM(0, pos=(0, 300))
g = geometry.square(numerical_aperture)
g[1, 1, 1] = 0
# p1 = Patch(s1, g)
s1.primary_phase_patch.geometry = g
pf = Patch(s1)
p2 = Patch(s2)
p3 = Patch(s2, geometry.square(0.2))
p4 = Patch(s2, geometry.square(0.1))
p3.phases = 0.25
p4.phases = 1
p4.additive_blend = False

pf.phases = patterns.lens(100, f=1 * u.m, wavelength=0.8 * u.um, extent=(10 * u.mm, 10 * u.mm))
rng = np.random.default_rng()
for n in range(50):
    random_data = rng.random([10, 10], np.float32) * 2.0 * np.pi
    s1.set_phases(random_data)
    s2.set_phases(n / 4.0)

p1 = None  # test deletion. After deleting the two windowed SLMs, we can create a new full screen one
s1.patches.clear()
s1 = 0
s2 = 0
s3 = SLM(1)  # full-screen window
s3.update()
s3.monitor_id = 0  # test switching monitor id
time.sleep(0.5)

s = SLM(0)
s.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255
s.wavelength = 0.804 * u.um
s.phases = (data.camera() / 255) * 2 * np.pi

s.update()
time.sleep(1)
s.wavelength = 503 * u.nm
s.update()
time.sleep(1)
