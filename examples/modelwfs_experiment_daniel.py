import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi
from openwfs.devices import LaserScanning, Gain
from openwfs.slm import SLM, Patch
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt
import hdf5storage as h5

# Define NI-DAQ channels for scanning
scanner = LaserScanning(
    x_mirror_mapping='Dev4/ao2',
    y_mirror_mapping='Dev4/ao3',
    input_mapping='Dev4/ai24',
    measurement_time=1000 * u.ms,
    delay=75,           # Unit? Assertion? Determine automatically with autocorr?
    height=512,
    width=512,
    zoom=6)

roi_detector = SingleRoi(scanner, x=50, y=50, radius=1)

# Define NI-DAQ Gain channel
gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0")
gain.on_reset(1)

gain.gain = 0.65    # PMT Gain in Volt


slm = SLM(2)


# hardcode offset, because our calibrations don't work yet
transform_scale_factor = 1.0
transform_matrix = np.array(fill_transform(slm, type='short')) * transform_scale_factor
# transform_matrix[2, :] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right
transform_matrix[2, :] = [0.0, 0.0, 1] # from the old hardcoded offset, visually adjusted to be right

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255 # again copied from earlier hardcodes
slm.wavelength = 0.808 * u.um

slm.transform = transform_matrix

# Load pattern on SLM
slmpatch = Patch(slm)

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()

input('Please turn on unshut laser and press Enter to continue...')

# Load data
patterndict = h5.loadmat(file_name="//ad.utwente.nl/tnw/BMPI/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-top-λ808.0nm.mat")
slmpatch.phases = -patterndict['phase_SLM']
slm.update()

# Perform scan
scanner.trigger()
img_raw = scanner.read()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()

# Show the scanned image
img = (2**15 - 1) - img_raw.astype('float32')
plt.imshow(img, vmin=0, vmax=1e3)
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads near top of dried up glass tube\nModel-based top correction, No FourierWFS')
plt.colorbar()
plt.show()
