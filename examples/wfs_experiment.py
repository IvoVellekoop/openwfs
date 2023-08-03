import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi
from openwfs.devices import LaserScanning
from openwfs.slm import SLM
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt

scanner = LaserScanning(x_mirror_mapping='Dev2/ao0', y_mirror_mapping='Dev2/ao1', input_mapping='Dev2/ai0',measurement_time=100 * u.ms)
roi_detector = SingleRoi(scanner, x=50, y=50, radius=1)


slm = SLM(2)

# hardcode offset, because our calibrations don't work yet
transform_matrix = np.array(fill_transform(slm,type='short'))
transform_matrix = transform_matrix*0.8 #scaling according to last
transform_matrix[2,:] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right

slm.transform = transform_matrix

controller = Controller(detector=roi_detector, slm=slm)
#alg = StepwiseSequential(n_x=1, n_y=1, phase_steps=3, controller=controller)
alg = CharacterisingFDR(max_modes=10,controller=controller)

t = alg.execute()
optimised_wf = np.angle(t)
slm.phases = optimised_wf
optimised_wf[optimised_wf < 0] += 2*np.pi #because the slm shows 0 to 2pi instead of -pi to pi
slm.update()
plt.imshow(optimised_wf,origin='lower')
plt.show()
