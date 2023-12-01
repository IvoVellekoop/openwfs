import numpy as np
import hdf5storage as h5
import astropy.units as u

from openwfs.slm import SLM, Patch
from openwfs.slm.geometry import fill_transform

# === SLM settings === #
slm = SLM(2)

# hardcode offset, because our calibrations don't work yet
transform_scale_factor = 1.032
transform_matrix = np.array(fill_transform(slm, fit='short')) * transform_scale_factor
# transform_matrix[2, :] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right
transform_matrix[2, :] = [0.0, 0.0, 1] # from the old hardcoded offset, visually adjusted to be right

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255     # again copied from earlier hardcodes
slm.wavelength = 0.808 * u.um

slm.transform = transform_matrix

slmpatch = Patch(slm)

# === Random SLM === #
# pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()

# === Flat SLM === #
# Flat SLM pattern
slmpatch.phases = 0
slm.update()

# === Model-based WFS === #
# Load data
patterndict_bottom_730 = h5.loadmat(
    file_name=r"\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\Chip-Laura\pattern-chip-laura-bottom-edge-wl730nm-cff0.35.mat")

system_aberration_pattern_dict = h5.loadmat(
    file_name=r"\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\28-Sep-2023-system-aberration\tube_ao_739157.585029_system-aberration\tube_ao_739157.585029_system-aberration_optimal_pattern.mat");

system_aberration_pattern = system_aberration_pattern_dict['slm_pattern_2pi_optimal']

slmpatch.phases = -patterndict_bottom_730['phase_SLM']
slm.update()

pass
