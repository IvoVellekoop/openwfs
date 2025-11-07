"""Micro-Manager simulated scanning microscope
======================================================================
This script simulates a scanning microscope with a pre-set image as a mock specimen.

To use it:

  * make sure  you have the PyDevice adapter installed in Micro-Manager (install the nightly build if you don't have it).
  * load the micro_manager_scanning_microscope.cfg hardware configuration in Micro-Manager,
  * locate the micro_manager_scanning_microscope.py in the file open dialog box that popps up.
  * take a snapshot or turn on live preview, you may need to auto-adjust the color scale
  * experiment with the scanning microscope parameters such as zoom in the script to see how they affect the image
"""

import astropy.units as u
import skimage

from openwfs.devices import ScanningMicroscope, Axis
from openwfs.devices.galvo_scanner import InputChannel

# parameters for Thorlabs SS30X-AG
scale = Axis.compute_scale(
    optical_deflection=1.0 / (0.22 * u.V / u.deg),
    galvo_to_pupil_magnification=2,
    objective_magnification=16,
    reference_tube_lens=200 * u.mm,
)

acceleration = Axis.compute_acceleration(
    optical_deflection=1.0 / (0.22 * u.V / u.deg),
    torque_constant=2.8e5 * u.dyne * u.cm / u.A,
    rotor_inertia=8.25 * u.g * u.cm**2,
    maximum_current=4 * u.A,
)

# scale = 440 * u.um / u.V (calibrated)
sample_rate = 0.5 * u.MHz
reference_zoom = 1.2
y_axis = Axis(
    channel="Dev4/ao0",
    v_min=-2.0 * u.V,
    v_max=2.0 * u.V,
    maximum_acceleration=acceleration,
    scale=scale,
)
x_axis = Axis(
    channel="Dev4/ao1",
    v_min=-2.0 * u.V,
    v_max=2.0 * u.V,
    maximum_acceleration=acceleration,
    scale=scale,
)
input_channel = InputChannel("Dev4/ai0", -1.0 * u.V, 1.0 * u.V)
test_image = skimage.data.hubble_deep_field() * 256

scanner = ScanningMicroscope(
    sample_rate=sample_rate,
    input=input_channel,
    y_axis=y_axis,
    x_axis=x_axis,
    test_pattern="image",
    reference_zoom=reference_zoom,
    resolution=1024,
    test_image=test_image,
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scanner.binning = 4
    plt.imshow(scanner.read(), cmap="gray")
    plt.colorbar()
    plt.show()
else:
    devices = {"microscope": scanner}
