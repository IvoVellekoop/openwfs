import numpy as np
def defocus(N_pixels):
    """Constructs a texture that represents a defocus: sqrt(1-r^2)"""

    # construct coordinate range. The full texture spans the range -1 to 1, and it is divided into N_pixels pixels.
    # The coordinates correspond to the centers of these pixels
    dx = 2.0 / N_pixels
    range_sqr = np.arange(-1.0 + 0.5 * dx, 1.0, dx) ** 2
    r_sqr = (1.0 - range_sqr.reshape(N_pixels, 1)) - range_sqr.reshape(1, N_pixels)
    return np.sqrt(np.maximum(r_sqr, 0.0))
