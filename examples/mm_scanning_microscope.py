"""
Constructs a scanning microscope controller for use with Micro-Manager

The microscope object can be loaded into Micro-Manager through the PyDevice
device adapter.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import skimage

# add 'openwfs' to the search path. This is only needed when developing openwfs
# otherwise it is just installed as a package
import set_path  # noqa
from openwfs.devices import ScanningMicroscope, Axis
from openwfs.devices.galvo_scanner import InputChannel

# parameters for Thorlabs SS30X-AG
scale = Axis.compute_scale(
    optical_deflection=1.0 / (0.22 * u.V / u.deg),
    galvo_to_pupil_magnification=2,
    objective_magnification=16,
    reference_tube_lens=200 * u.mm)

acceleration = Axis.compute_acceleration(
    optical_deflection=1.0 / (0.22 * u.V / u.deg),
    torque_constant=2.8E5 * u.dyne * u.cm / u.A,
    rotor_inertia=8.25 * u.g * u.cm ** 2,
    maximum_current=4 * u.A)

# scale = 440 * u.um / u.V (calibrated)
sample_rate = 0.5 * u.MHz
reference_zoom = 1.2
y_axis = Axis(channel='Dev4/ao0', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=acceleration, scale=scale)
x_axis = Axis(channel='Dev4/ao1', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=acceleration, scale=scale)
input_channel = InputChannel('Dev4/ai0', -1.0 * u.V, 1.0 * u.V)
test_image = skimage.data.hubble_deep_field() * 256

scanner = ScanningMicroscope(sample_rate=sample_rate,
                             input=input_channel, y_axis=y_axis, x_axis=x_axis,
                             test_pattern='image', reference_zoom=reference_zoom,
                             resolution=1024, test_image=test_image)

if __name__ == '__main__':
    scanner.binning = 4
    plt.imshow(scanner.read(), cmap='gray')
    plt.colorbar()
    plt.show()
else:
    devices = {'microscope': scanner}
