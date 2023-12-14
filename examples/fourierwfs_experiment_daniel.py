from time import sleep
import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.processors import SingleRoi
from openwfs.devices import LaserScanning, Gain
from openwfs.slm import SLM, Patch
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt
import hdf5storage as h5


vmax = 400

# Define NI-DAQ channels for scanning
scanner = LaserScanning(
    x_mirror_mapping='Dev4/ao2',
    y_mirror_mapping='Dev4/ao3',
    input_mapping='Dev4/ai24',
    duration=700 * u.ms,
    delay=0,           # Unit? Assertion? Determine automatically with autocorr?
    data_shape=(256, 256),
    zoom=40)

scanner_big = LaserScanning(
    x_mirror_mapping='Dev4/ao2',
    y_mirror_mapping='Dev4/ao3',
    input_mapping='Dev4/ai24',
    duration=1400 * u.ms,
    delay=0,           # Unit? Assertion? Determine automatically with autocorr?
    data_shape=(512, 512),
    zoom=40)

roi_detector = SingleRoi(scanner, x=128, y=128, radius=127)

# Define NI-DAQ Gain channel
gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0")

# Set gain
# gain.on_reset(1)
# gain.gain = 0.7    # PMT Gain in Volt

# === SLM settings === #
slm = SLM(2)

# hardcode offset, because our calibrations don't work yet
transform_scale_factor = 1.032
transform_matrix = np.array(fill_transform(slm, fit='short')) * transform_scale_factor
# transform_matrix[2, :] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right
transform_matrix[2, :] = [0.0, 0.0, 1]

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255     # again copied from earlier hardcodes
slm.wavelength = 0.804 * u.um

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
img_raw_flat_init = scanner.read().copy()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()

plt.figure()
img_flat_init = img_raw_flat_init.astype('float32') - (2 ** 15 - 1)
plt.imshow(img_flat_init, vmin=0, vmax=vmax, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Content with image?')
plt.colorbar()
plt.show()


# === FourierWFS === #
# FourierWFS settings
alg = BasicFDR(
    feedback=roi_detector,
    slm=slm,
    k_angles_min=-2,
    k_angles_max=2,
    overlap=0.1,
    phase_steps=8,
    slm_shape=(1152, 1152))

# Flat SLM
slmpatch.phases = 0
slm.update()

# Execute FourierWFS
t = alg.execute()
optimised_wf = np.angle(t)

# With correction
slm.patches[0].phases = 0
slmpatch.phases = optimised_wf
slm.update()

img_raw_fourierwfs = scanner_big.read().copy()


# With minus correction
slmpatch.phases = -optimised_wf
slm.update()

img_raw_fourierwfs_min = scanner_big.read().copy()

# === Flat SLM === #
# Flat SLM pattern
slmpatch.phases = 0
slm.update()

# Perform scan
img_raw_flat = scanner_big.read().copy()

# Random SLM pattern to destroy focus and prevent bleaching
slmpatch.phases = np.random.rand(300, 300) * 2 * np.pi
slm.update()


# === Plot results === #

plt.figure()
img_flat_init = img_raw_flat_init.astype('float32') - (2 ** 15 - 1)
plt.imshow(img_flat_init, vmin=0, vmax=vmax, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Initial image')
plt.colorbar()

# Show the scanned image
plt.figure()
img_flat = img_raw_flat.astype('float32') - (2 ** 15 - 1)
plt.imshow(img_flat, vmin=0, vmax=vmax, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads\nno correction')
plt.colorbar()

# Show the scanned image
plt.figure()
img_fourierwfs = img_raw_fourierwfs.astype('float32') - (2 ** 15 - 1)
plt.imshow(img_fourierwfs, vmin=0, vmax=vmax, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads, with FourierWFS')
plt.colorbar()

# Show the scanned image
plt.figure()
img_fourierwfs_min = img_raw_fourierwfs_min.astype('float32') - (2 ** 15 - 1)
plt.imshow(img_fourierwfs_min, vmin=0, vmax=vmax, interpolation='nearest')
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title('Beads, with FourierWFS minus')
plt.colorbar()

t_square = t[:, :]

# Show patterns
plt.figure()
plt.imshow(np.angle(t_square), interpolation='nearest')
plt.title('FourierWFS correction')
plt.colorbar()
plt.show()

pass
