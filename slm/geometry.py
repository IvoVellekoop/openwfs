import numpy as np


def rectangle(left, top, right, bottom):
    return np.array([
        [[left, top, 0.0, 0.0], [right, top, 1.0, 0.0]],
        [[left, bottom, 0.0, 1.0], [right, bottom, 1.0, 1.0]]],
        dtype=np.float32)


def square(radius):
    return rectangle(-radius, -radius, radius, radius)
