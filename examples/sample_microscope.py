import numpy as np
import astropy.units as u
import set_path
from openwfs.simulation import Microscope, MockImageSource

img = np.maximum(np.random.randint(-10000, 100, (500, 500), dtype=np.int16), 0) + 20
src = MockImageSource.from_image(img, 100 * u.nm)
mic = Microscope(src, m=10, na=0.85, wavelength=532.8 * u.nm, pixel_size=6.45 * u.um)
devices = {'camera': mic.camera, 'stage': mic.stage}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = mic.camera
    plt.ion()  # turn on interactive mode
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.show()
    for p in range(100):
        mic.stage.position_x = p * 1 * u.um
        c.trigger()
        cim = c.read()
        plt.imshow(cim)
        plt.draw()
        plt.pause(0.2)
