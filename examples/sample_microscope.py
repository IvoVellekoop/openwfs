import numpy as np
import astropy.units as u
import set_path  # noqa - needed for setting the module search path to find openwfs
from openwfs.simulation import Microscope, MockSource
from openwfs.utilities import grab_and_show, Transform

### Parameters that can be altered

img_size_x = 1024
# Determines how wide the image is.

img_size_y = 1024
# Determines how high the image is.

magnification = 40
# magnification from object plane to camera.

numerical_aperture = 0.85
# numerical aperture of the microscope objective

wavelength = 532.8 * u.nm
# wavelength of the light, different wavelengths are possible, units can be adjusted accordingly.

pixel_size = 6.45 * u.um
# Size of the pixels on the camera

camera_resolution = (256, 256)
# number of pixels on the camera

p_limit = 100
# Number of iterations. Influences how quick the 'animation' is complete.

## Code
img = np.maximum(np.random.randint(-10000, 100, (img_size_y, img_size_x), dtype=np.int16), 0)
src = MockSource(img, 50 * u.nm)
mic = Microscope(src, magnification=magnification, numerical_aperture=numerical_aperture, wavelength=wavelength,
                 truncation_factor=0.5)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = mic.get_camera(shot_noise=True, digital_max=255, transform=Transform.zoom(2.0))
devices = {'camera': cam, 'stage': mic.xy_stage}

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original image')
    ax = plt.subplot(1, 2, 2)
    plt.title('Scanned image')
    for p in range(p_limit):
        # mic.xy_stage.x = p * 1 * u.um
        mic.numerical_aperture = 1.0 * (p + 1) / p_limit  # NA increases to 1.0
        grab_and_show(cam)
        plt.title(f"NA: {mic.numerical_aperture}, δ: {mic.abbe_limit.to_value(u.um):2.2} μm")
        plt.pause(0.2)
