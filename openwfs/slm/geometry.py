import numpy as np


def rectangle(left, top, right, bottom):
    return np.array([
        [[left, top, 0.0, 0.0], [right, top, 1.0, 0.0]],
        [[left, bottom, 0.0, 1.0], [right, bottom, 1.0, 1.0]]],
        dtype=np.float32)


def square(radius):
    return rectangle(-radius, -radius, radius, radius)


def fill_transform(slm, fit='short'):
    """Constructs a transformation matrix that makes a 'square' patch (range -1.0 to 1.0 in x and y) fill the SLM.
    :param slm:     SLM object, only used to obtain width and height
    :param fit:    'full', make the patch fill the full SLM, causes the pixels in a square texture to become
                        stretched (non-square)
                    'short', make the patch square, filling the short side of the SLM.
                    'small', make the patch square, filling the long side of the SLM. If the SLM is not square, part of
                    the patch will not be displayed because it is outside the SLM window.
    """
    width = slm.shape[1]
    height = slm.shape[0]
    if fit == 'full':
        return np.eye(3)
    elif fit != 'short' and fit != 'long':
        raise ValueError("Unsupported type")

    if (width > height) == (fit == 'short'):  # scale width?
        return [[height / width, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else:  # or scale height
        return [[1.0, 0.0, 0.0], [0.0, width / height, 0.0], [0.0, 0.0, 1.0]]
