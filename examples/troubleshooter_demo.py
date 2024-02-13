import numpy as np
import skimage
import astropy.units as u

from openwfs.processors import SingleRoi
from openwfs.simulation import SimulatedWFS, MockSource, MockSLM, Microscope
from openwfs.algorithms import StepwiseSequential
from openwfs.algorithms.troubleshoot import troubleshoot, field_correlation

from openwfs.core import Device

Device.multi_threading = False  # for easier debugging


def phase_response_test_function(phi, b, c, gamma):
    """A synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return np.clip(2 * np.pi * (b + c * (phi / (2 * np.pi)) ** gamma), 0, None)


def inverse_phase_response_test_function(f, b, c, gamma):
    """Inverse of the synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return 2 * np.pi * ((f / (2 * np.pi) - b) / c) ** (1 / gamma)


def lookup_table_test_function(f, b, c, gamma):
    """
    Compute the lookup indices (i.e. a lookup table)
    for countering the synthetic phase response test function: 2π*(b + c*(phi/2π)^gamma).
    """
    phase = inverse_phase_response_test_function(f, b, c, gamma)
    return (np.mod(phase, 2 * np.pi) * 256 / (2 * np.pi) + 0.5).astype(np.uint8)


# === Define mock hardware ===
n_x = 10
n_y = 10
num_phase_steps = 8

# Aberration and image source
gaussian_noise_std = 1.0
numerical_aperture = 1.0
aberration_phase = np.random.uniform(0.0, 2 * np.pi, (n_y, n_x))
aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)
img_off = np.zeros((120, 120), dtype=np.int16)
img_on = img_off.copy()
img_on[60, 60] = 1000
img_on[60, 70] = 1000
img_on[60, 80] = 1000
img_on[60, 90] = 1000
img_on[60, 100] = 1000
img_on[60, 110] = 1000
src = MockSource(img_on, pixel_size=200 * u.nm)

# SLM with incorrect phase response
slm = MockSLM(shape=(100, 100))
# linear_phase = np.arange(0, 2*np.pi, 2*np.pi/256)
# slm.phase_response = phase_response_test_function(linear_phase, b=0.02, c=0.9, gamma=1.2)

# Simulation with noise, camera, ROI detector
sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=numerical_aperture,
                 aberrations=aberration, wavelength=800 * u.nm)
cam = sim.get_camera(analog_max=2e4, gaussian_noise_std=gaussian_noise_std)
roi_detector = SingleRoi(cam, radius=1)  # Only measure that specific point

# === Define and run WFS algorithm ===
alg = StepwiseSequential(feedback=roi_detector, slm=slm, n_x=n_x, n_y=n_y, phase_steps=num_phase_steps)

trouble = troubleshoot(algorithm=alg,
                       frame_source=cam,
                       laser_block=cam.laser_block, laser_unblock=cam.laser_unblock,
                       stability_sleep_time_s=0.1, stability_num_of_frames=30)
cam.laser_block()

trouble.report(do_plots=False)

print(f'{np.abs(field_correlation(trouble.wfs_result.t, np.exp(1j * aberration_phase))) ** 2:.4f}')

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(trouble.after_frame, vmin=0, vmax=500)
plt.title('After')
plt.colorbar()

plt.figure()
plt.imshow(trouble.shaped_wf_frame, vmin=0, vmax=500)
plt.title('Shaped WF')
plt.colorbar()

plt.figure()
cam.laser_unblock()
slm.set_phases(-aberration_phase)
cam.read()
slm.set_phases(-aberration_phase)
cam.read()
slm.set_phases(-aberration_phase)
cam.read()
plt.imshow(cam.read(), vmin=0, vmax=500)
plt.title('Perfect WF')
plt.colorbar()

plt.show()
