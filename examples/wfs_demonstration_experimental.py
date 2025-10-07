"""
WFS demo experiment
=====================
This script demonstrates how to perform a wavefront shaping experiment using the openwfs library.
It performs wavefront shaping with a FourierDualReference algorithm, maximizing feedback from a
single region of interest (ROI) on a camera.
It assumes that you have a genicam-compatible camera and an SLM connected to the video output of your computer.

Adjustments to make for your specific setup:
* the path to the camera driver.
* The code assumes the SLM is connected as a secondary monitor. If not, adjust the monitor_id below.
* The gray value that corresponds to a 2pi phase shift on the SLM. This is used to set the lookup table of the SLM.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from openwfs.algorithms import FourierDualReference
from openwfs.devices import Camera, SLM
from openwfs.processors import SingleRoi
from openwfs.utilities import Transform

# Adjust these parameters to your setup
# The camera driver file path
camera_driver_path = R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti"
monitor_id = 2
# time to stabilize the SLM, in frames
settle_time = 2
# we are using a setup with an SLM that produces 2pi phase shift
# at a gray value of 142. With proper calibration, this is usually 256.
two_pi_gray_value = 142
centering = Transform(source_origin=(0, 0), destination_origin=(0.0, 0.05))


# This script shows how a wavefront shaping experiment can be performed from Python
cam = Camera(camera_driver_path)
cam.exposure = 16.666 * u.ms
roi_detector = SingleRoi(cam, radius=2)

# constructs the actual slm for wavefront shaping, and a monitor window to display the current phase pattern
slm = SLM(monitor_id=monitor_id, duration=settle_time, transform=centering)
monitor = slm.clone(monitor_id=0, pos=(0, 0), shape=(slm.shape[0] // 3, slm.shape[1] // 3))

slm.lookup_table = range(two_pi_gray_value)
alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=[800, 800], k_radius=7)

result = alg.execute()
print(result)
optimised_wf = -np.angle(result.t)
before = roi_detector.read()
slm.set_phases(optimised_wf)
after = roi_detector.read()
print(f"actual_optimized_intensity: {after}")
print(f"improvement_ratio: {after / before}")

while True:
    slm.set_phases(optimised_wf)
    plt.pause(1.0)
    slm.set_phases(0.0)
    plt.pause(1.0)
