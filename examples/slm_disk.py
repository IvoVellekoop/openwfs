"""
SLM disk
=============
This script demonstrates how to create a circular geometry on an SLM
and superpose a gradient pattern on it.
"""

import cv2
import numpy as np

from openwfs.devices.slm import SLM, Patch, geometry
from openwfs.utilities import patterns

# construct a windowed-mode, square SLM window
slm_size = (400, 400)
slm = SLM(monitor_id=0, shape=slm_size)

# for the first patch, use a circular geometry, where a 1-D texture is mapped
# onto a set of concentric rings. Display a gradient pattern
shape = geometry.circular(radii=(0, 0.4, 0.7, 1.0), segments_per_ring=(4, 6, 8))
slm.patches[0].geometry = shape
phases = np.random.uniform(low=0, high=30, size=(1, 18))
slm.patches[0].set_phases(phases, update=False)

# add a second patch that corresponds to a linear gradient
gradient = patterns.tilt(slm_size, (10, 25))
slm.patches.append(Patch(slm))
slm.patches[1].set_phases(gradient)

# read back the pixels and store in a file
pixels = slm.pixels.read()
cv2.imwrite("slm_disk.png", pixels)
