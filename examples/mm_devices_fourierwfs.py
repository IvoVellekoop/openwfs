import set_path
import numpy as np
from openwfs.algorithms import FourierDualReference
from openwfs.algorithms.utilities import WFSController
from openwfs.processors import SingleRoi
from openwfs.devices import ScanningMicroscope, Gain
from openwfs.slm import SLM, Patch
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration

"""
Script for setting up a microscope, gain and WFS objects.
These are put into the devices dict such that they can be read by PyDevice.
"""

max_FOV_V = (1.0/30) * u.V

# Define NI-DAQ channels and settings for scanning
scanner = ScanningMicroscope(
    bidirectional=True,
    sample_rate=0.2 * u.MHz,
    axis0=('Dev4/ao2', -max_FOV_V, max_FOV_V),
    axis1=('Dev4/ao3', -max_FOV_V, max_FOV_V),
    input=('Dev4/ai16', -1.0 * u.V, 1.0 * u.V, TerminalConfiguration.DIFF),
    data_shape=(240, 240),
    scale=440 * u.um / u.V,
    delay=58.0,
    padding=0.05)


# Define NI-DAQ Gain channel and settings
gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0",
    reset=False,
    gain=0.65 * u.V,
)


# ROI detector
roi_detector = SingleRoi(scanner, x=60, y=60, radius=29)


# SLM
slm = SLM(2, wavelength=804*u.nm)
slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255     # Temporary, hardcoded lookup table
transform_scale_factor = 1.032
transform_matrix = np.array(fill_transform(slm, fit='short')) * transform_scale_factor
transform_matrix[2, :] = [0.0, 0.0, 1]
slm.transform = transform_matrix


# Wavefront Shaping algorithm
wfs_alg = FourierDualReference(
    feedback=roi_detector,
    slm=slm,
    slm_shape=(1152, 1152),
    k_angles_min=-2,
    k_angles_max=2,
    phase_steps=6)


# WFS controller
wfs_controller = WFSController(wfs_alg)


# Devices
devices = {
    'scanner': scanner,
    'gain': gain,
    'roi_detector': roi_detector,
    'slm': slm,
    'wfs_controller': wfs_controller,
    'wfs_alg': wfs_alg,
}

