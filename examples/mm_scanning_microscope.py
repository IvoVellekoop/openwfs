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

scale = 440 * u.um / u.V
sample_rate = 0.5 * u.MHz
reference_zoom = 1.2
y_axis = Axis(channel='Dev4/ao0', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=10 * u.V / u.ms ** 2)
x_axis = Axis(channel='Dev4/ao1', v_min=-2.0 * u.V, v_max=2.0 * u.V, maximum_acceleration=10 * u.V / u.ms ** 2)
test_image = skimage.data.hubble_deep_field() * 256

scanner = ScanningMicroscope(sample_rate=sample_rate,
                             input=('Dev4/ai0', -1.0 * u.V, 1.0 * u.V), y_axis=y_axis, x_axis=x_axis,
                             scale=scale, test_pattern='image', reference_zoom=reference_zoom,
                             test_image=test_image)

if __name__ == '__main__':
    scanner.binning = 4
    plt.imshow(scanner.read(), cmap='gray')
    plt.colorbar()
    plt.show()
else:
    devices = {'microscope': scanner}
