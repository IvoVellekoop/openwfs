import numpy as np
from openwfs.algorithms import StepwiseSequential
from openwfs.processors import SingleRoi
from openwfs.devices import Camera
from openwfs.core import Device
from openwfs.slm import SLM
from openwfs.slm.geometry import fill_transform
import matplotlib.pyplot as plt

# This python script shows how a wavefront shaping experiment can be performed from Python
Device.multi_threading = False
cam = Camera()
cam.nodes.ExposureTime.value = 16666
roi_detector = SingleRoi(cam, radius=2)

slm = SLM(monitor_id=2, settle_time=2)

# hardcode offset, because our calibrations don't work yet
# transform_matrix = np.array(fill_transform(slm, fit='short'))
# transform_matrix = transform_matrix * 0.8  # scaling according to last
# transform_matrix[2, :] = [-0.0147 / (0.4 + 0.5), 0.0036 / 0.5,
#                         1]  # from the old hardcoded offset, visually adjusted to be right

# slm.transform = transform_matrix

alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=6, phase_steps=8)

result = alg.execute()
print(result)
print(result.estimated_enhancement)
print(result.noise_factor)
optimised_wf = -np.angle(result.t[:, :, 0])
before = roi_detector.read()
slm.set_phases(optimised_wf)
after = roi_detector.read()
slm.update()
plt.imshow(np.abs(result.t[:, :, 0]), origin='lower')
plt.show()
