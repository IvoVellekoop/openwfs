"""
Calibration demo experiment - Twymann-Green interferometer
=====================
This script performs an SLM field response calibration using Fourier fringe analysis and a Twymann-Green interferometer.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from openwfs.calibration import FringeAnalysisSLMCalibrator
from openwfs.devices import Camera, SLM
from openwfs.simulation import Camera as MockCam


repetitions = 10        # How often to run the experiment (for e.g. error bars)

# This script shows how a wavefront shaping experiment can be performed from Python
cam = Camera(R"C:\Program Files\Basler\pylon 6\Runtime\x64\ProducerU3V.cti")
cam.exposure_time = 90 * u.ms

# constructs the actual slm for wavefront shaping, and a monitor window to display the current phase pattern
slm = SLM(2)
monitor = slm.clone(monitor_id=0, pos=(0, 0), shape=(slm.shape[0] // 4, slm.shape[1] // 4))

# Define cropping area for modulated and reference fringes
s = cam.data_shape
modulated_slices = (slice(None), slice(s[0] // 6, s[0] // 3))
reference_slices = (slice(None), slice(-s[0] // 3, -s[0] // 6))

calibrator = FringeAnalysisSLMCalibrator(
    camera=cam, slm=slm, modulated_slices=modulated_slices, reference_slices=reference_slices
)

for r in range(repetitions):
    field, gray_values, frames = calibrator.execute()

    np.savez(
        f'C:/LocalData/tg-fringe-slm-calibration-r{r}.npz',
        frames=frames,
        field=field,
        gray_values=gray_values,
        modulated_slices=modulated_slices,
        reference_slices=reference_slices,
    )
