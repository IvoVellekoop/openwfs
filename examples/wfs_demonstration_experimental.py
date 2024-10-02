"""
WFS demo experiment
=====================
This script demonstrates how to perform a wavefront shaping experiment using the openwfs library.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from openwfs.algorithms import FourierDualReference
from openwfs.devices import Camera, SLM
from openwfs.processors import SingleRoi

# This script shows how a wavefront shaping experiment can be performed from Python
cam = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
cam.exposure_time = 16.666 * u.ms
roi_detector = SingleRoi(cam, radius=2)

# constructs the actual slm for wavefront shaping, and a monitor window to display the current phase pattern
slm = SLM(monitor_id=2, duration=2)
monitor = slm.clone(
    monitor_id=0, pos=(0, 0), shape=(slm.shape[0] // 4, slm.shape[1] // 4)
)

# we are using a setup with an SLM that produces 2pi phase shift
# at a gray value of 142
slm.lookup_table = range(142)
alg = FourierDualReference(
    feedback=roi_detector, slm=slm, slm_shape=[800, 800], k_radius=7
)

result = alg.execute()
print(result)
print(result.estimated_enhancement)
print(result.fidelity_noise)
optimised_wf = -np.angle(result.t)
before = roi_detector.read()
slm.set_phases(optimised_wf)
after = roi_detector.read()
print(f"actual_optimized_intensity: {after}")
print(f"improvement_ratio: {after / before}")

while True:
    slm.set_phases(optimised_wf)
    plt.pause(1.0)
    slm.set_phases(0.0)
    plt.pause(1.0)

# plt.show()
# input("press any key")
