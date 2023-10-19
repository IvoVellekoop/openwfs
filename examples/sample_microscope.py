import numpy as np
import astropy.units as u
import set_path
from openwfs.simulation import Microscope, MockImageSource

### Parameters that can be altered

magnification = 10
# magnification from object plane to camera.

numerical_aperture = 0.85
# numerical aperture of the microscope objective

wavelength = 532.8 * u.nm
# wavelength of the light, different wavelengths are possible, units can be adjusted accordingly.

pixel_size = 6.45 * u.um
# Size of the pixels on the camera object that represents the microscope image. Influences the size of the moving dots.

p_limit = 100
# Number of iterations. Influences how quick the 'animation' is complete.

## Code

img = np.maximum(np.random.randint(-10000, 100, (500, 500), dtype=np.int16), 0) + 20
src = MockImageSource.from_image(img, 100 * u.nm)
mic = Microscope(src, m=magnification, na=numerical_aperture, wavelength=wavelength, pixel_size=pixel_size)
mic.camera.saturation = 70.0
devices = {'camera': mic.camera, 'stage': mic.stage}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = mic.camera
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    ax = plt.subplot(1, 2, 2)
    for p in range(p_limit):
        mic.stage.x = p * 1 * u.um
        c.trigger()
        cim = c.read()
        ax.imshow(cim)
        plt.draw()
        plt.pause(0.2)
