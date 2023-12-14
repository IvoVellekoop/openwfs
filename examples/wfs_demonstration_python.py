import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.processors import SingleRoi
from openwfs.devices import ScanningMicroscope
from openwfs.slm import SLM
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt

# This python script shows how a wavefront shaping experiment can be performed from Python

scanner = ScanningMicroscope(x_mirror_mapping='Dev2/ao0', y_mirror_mapping='Dev2/ao1', input_mapping='Dev2/ai0',
                             measurement_time=100 * u.ms)
roi_detector = SingleRoi(scanner, x=scanner.data_shape[1], y=scanner.data_shape[1], radius=1)

slm = SLM(0, wavelength=804)
slm.lut_generator = lambda λ: np.arange(0,
                                        0.2623 * λ.to(u.nm).value - 23.33) / 255  # again copied from earlier hardcodes
slm.wavelength = 804 * u.nm

# hardcode offset, because our calibrations don't work yet
transform_matrix = np.array(fill_transform(slm, fit='short'))
transform_matrix = transform_matrix * 0.8  # scaling according to last
transform_matrix[2, :] = [-0.0147 / (0.4 + 0.5), 0.0036 / 0.5,
                          1]  # from the old hardcoded offset, visually adjusted to be right

slm.transform = transform_matrix

# alg = StepwiseSequential(n_x=1, n_y=1, phase_steps=3, controller=controller)
alg = CharacterisingFDR(feedback=roi_detector, slm=slm, max_modes=10)

t = alg.execute()
optimised_wf = np.angle(t)
slm.phases = optimised_wf
optimised_wf[optimised_wf < 0] += 2 * np.pi  # because the slm shows 0 to 2pi instead of -pi to pi
slm.update()
plt.imshow(optimised_wf, origin='lower')
plt.show()
