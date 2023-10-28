import numpy as np
import astropy.units as u
import set_path
from openwfs.simulation import Microscope, MockImageSource

### Parameters that can be altered

img_size_x = 500
# Determines how wide the image is.

img_size_y = 500
# Determines how high the image is.

magnification = 10
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
src = MockImageSource.from_image(img, 100 * u.nm)
mic = Microscope(src, magnification=magnification, numerical_aperture=numerical_aperture, wavelength=wavelength,
                 camera_pixel_size=pixel_size, camera_resolution=camera_resolution)
mic.camera.saturation = 70.0
devices = {'camera': mic.camera, 'stage': mic.xy_stage}

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = mic.camera
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original image')
    ax = plt.subplot(1, 2, 2)
    plt.title('Scanned image')
    for p in range(p_limit):
        mic.xy_stage.x = p * 1 * u.um
        mic.numerical_aperture = 1.0  # * (p + 1) / p_limit  # NA increases to 1.0
        c.trigger()
        cim = c.read()
        ax.imshow(cim)
        plt.draw()
        plt.pause(0.2)
