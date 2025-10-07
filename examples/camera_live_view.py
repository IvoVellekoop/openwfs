"""
Camera live view
=====================
This script gives a _very_ rudimentary live viewer for a camera.
"""

import astropy.units as u
import matplotlib.pyplot as plt

from openwfs.devices import Camera

# Adjust these parameters to your setup
# The camera driver file path
camera_driver_path = R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti"
cam = Camera(camera_driver_path)
cam.exposure = 16.666 * u.ms

frame = 0
plt.figure()
im = plt.imshow(cam.read())
plt.colorbar()
plt.show(block=False)
while True:
    im.set_data(cam.read())
    plt.title(f"frame: {frame}")
    plt.show(block=False)
    plt.pause(0.001)
