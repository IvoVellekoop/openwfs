import numpy as np
from openwfs.algorithms import FourierDualReference
from openwfs.processors import SingleRoi
from openwfs.devices import Camera
from openwfs.core import Device
from openwfs.slm import SLM
import matplotlib.pyplot as plt
import astropy.units as u

# This script shows how a wavefront shaping experiment can be performed from Python
cam = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
cam.exposure_time = 16.666 * u.ms
roi_detector = SingleRoi(cam, radius=2)

# constructs the actual slm for wavefront shaping, and a monitor window to display the current phase pattern
slm = SLM(monitor_id=2, duration=2)
# we are using a setup with an SLM that is not calibrated and produces 2pi phase shift
# at a gray value of 142
slm.lookup_table = range(142)
alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=[800, 800], k_angles_min=-5, k_angles_max=5)

result = alg.execute()
print(result)
print(result.estimated_enhancement)
print(result.noise_factor)
optimised_wf = -np.angle(result.t)
before = roi_detector.read()
slm.set_phases(optimised_wf)
after = roi_detector.read()
print(f"actual_optimized_intensity: {after}")
print(f"improvement_ratio: {after / before}")
plt.imshow(np.abs(result.t), origin='lower')
# plt.show()
# input("press any key")
