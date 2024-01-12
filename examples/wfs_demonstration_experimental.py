import numpy as np
from openwfs.algorithms import StepwiseSequential
from openwfs.processors import SingleRoi
from openwfs.devices import Camera
from openwfs.core import Device
from openwfs.slm import SLM
import matplotlib.pyplot as plt

# This script shows how a wavefront shaping experiment can be performed from Python
Device.multi_threading = False
cam = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
cam.nodes.ExposureTime.value = 16666
roi_detector = SingleRoi(cam, radius=2)

slm = SLM(monitor_id=2, duration=2)
slm.lookup_table = range(142)
alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=6, phase_steps=9)

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
plt.show()
input("press any key")
