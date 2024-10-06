import numpy as np


def complex_random(shape):
    """
    Create an array with normally distributed complex elements with variance 1.
    """
    return np.sqrt(0.5) * (np.random.normal(size=shape) + 1j * np.random.normal(size=shape))
