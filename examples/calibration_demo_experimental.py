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

# This script shows how a wavefront shaping experiment can be performed from Python
cam = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
cam.exposure_time = 16.666 * u.ms

# constructs the actual slm for wavefront shaping, and a monitor window to display the current phase pattern
slm = SLM(2)
monitor = slm.clone(monitor_id=0, pos=(0, 0), shape=(slm.shape[0] // 4, slm.shape[1] // 4))

calibrator = FringeAnalysisSLMCalibrator(camera=cam, slm=slm)
field, gray_values, frames = calibrator.execute()

plt.figure()
phase = np.unwrap(np.angle(field))
plt.plot(phase)
plt.show()
