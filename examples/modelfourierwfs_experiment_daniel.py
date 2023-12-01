import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.processors import SingleRoi
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
    duration=400 * u.ms,
    delay=62,           # Unit? Assertion? Determine automatically with autocorr?
    height=256,
    width=256,
    zoom=4)

roi_detector = SingleRoi(scanner, x=128, y=128, radius=127)

# Define NI-DAQ Gain channel
gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0")

# Set gain
gain.on_reset(1)
gain.gain = 0.7    # PMT Gain in Volt

# === SLM settings === #
slm = SLM(2)

# hardcode offset, because our calibrations don't work yet
transform_scale_factor = 1.032
transform_matrix = np.array(fill_transform(slm, type='short')) * transform_scale_factor
# transform_matrix[2, :] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right
transform_matrix[2, :] = [0.0, 0.0, 1]

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255     # again copied from earlier hardcodes
slm.wavelength = 0.808 * u.um

slm.transform = transform_matrix

slmpatch = Patch(slm)

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()


# === Start measurements === #
input('\nPlease unshut laser and press Enter to continue...')


# === Flat SLM === #
# Flat SLM pattern
slmpatch.phases = 0
slm.update()

# Perform scan
scanner.trigger()
img_raw_flat = scanner.read().copy()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()


# === Model-based WFS === #
# Load data
patterndict = h5.loadmat(file_name=r"\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\slm-patterns\Chip-Laura\pattern-chip-laura-top-edge-wl730nm-cff0.35.mat")
slmpatch.phases = -patterndict['phase_SLM']
slm.update()

# Perform scan
scanner.trigger()
img_raw_modelwfs1 = scanner.read().copy()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()

# Show the scanned image
img_modelwfs1 = (2**15 - 1) - img_raw_modelwfs1.astype('float32')
plt.imshow(img_modelwfs1, vmin=0, vmax=1e3, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Bottom of tube\nModel-based top correction, No FourierWFS')
plt.colorbar()
plt.show()


# === FourierWFS === #
# FourierWFS settings
controller = Controller(detector=roi_detector, slm=slm)
alg = BasicFDR(k_angles_min=-2, k_angles_max=2, overlap=0.1, phase_steps=6)

# Put model-based pattern on SLM
slmpatch.phases = -patterndict['phase_SLM']
slm.update()

# Execute FourierWFS
t = alg.execute()
optimised_wf = np.angle(t)
slm.phases = optimised_wf
slm.update()

# Perform scan with model+fourier wfs
scanner.trigger()
img_raw_modelfourierwfs = scanner.read().copy()


# === Model-based WFS === #
# Load data
slmpatch.phases = -patterndict['phase_SLM']
slm.update()

# Perform scan
scanner.trigger()
img_raw_modelwfs2 = scanner.read().copy()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()


# === Plot results === #
# Show the scanned image
plt.figure()
img_flat = (2**15 - 1) - img_raw_flat.astype('float32')
plt.imshow(img_flat, vmin=0, vmax=2e3, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads near top of dried up glass tube\nno correction')
plt.colorbar()

# Show the scanned image
plt.figure()
img_modelwfs2 = (2**15 - 1) - img_raw_modelwfs2.astype('float32')
plt.imshow(img_modelwfs2, vmin=0, vmax=2e3, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads near top of dried up glass tube\nwith model-based SA correction, no FourierWFS')
plt.colorbar()

# Show the scanned image
plt.figure()
img_modelfourierwfs = (2**15 - 1) - img_raw_modelfourierwfs.astype('float32')
plt.imshow(img_modelfourierwfs, vmin=0, vmax=2e3, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Bottom of tube\nwith model-based SA correction, with FourierWFS')
plt.colorbar()

t_square = t[:, 384:1536]

# Show patterns
plt.figure()
plt.imshow(np.mod(-patterndict['phase_SLM'], 2*np.pi), interpolation='nearest')
plt.title('ModelWFS correction')
plt.colorbar()

plt.figure()
plt.imshow(np.angle(t_square), interpolation='nearest')
plt.title('FourierWFS correction')
plt.colorbar()

plt.figure()
plt.imshow(np.mod(-patterndict['phase_SLM'] + np.angle(t_square), 2*np.pi), interpolation='nearest')
plt.title('Model+FourierWFS correction')
plt.colorbar()

plt.show()

pass
