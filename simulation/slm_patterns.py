import numpy as np


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def generate_double_pattern(shape, phases_half1, phases_half2, phase_offset):
    width, height = shape
    half_width = width // 2

    # Generate halves the image with phase offset
    if phases_half1 == 0:
        half1 = [[0 for _ in range(half_width)] for _ in range(height)]
    else:
        half1 = [[int(((i + phase_offset) / (width / phases_half1)) * 256) % 256 for _ in range(half_width)] for i in
                 range(height)]

    if phases_half2 == 0:
        half2 = [[0 for _ in range(half_width)] for _ in range(height)]
    else:
        half2 = [[int((i / (width / phases_half2)) * 256) % 256 for _ in range(half_width)] for i in range(height)]

    # Combine both halves
    image_array = [row1 + row2 for row1, row2 in zip(half1, half2)]

    # Resize the image to the specified shape
    image_array = [row[:width] for row in image_array[:height]]

    return np.array(image_array)
