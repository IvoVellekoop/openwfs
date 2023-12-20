import numpy as np


def rectangle(left, top, right, bottom):
    """
    Creates a rectangle shape defined by its corner coordinates in a normalized coordinate system.
    The rectangle is represented as a 2x2x4 array where each inner array represents a vertex.
    The vertices are specified in counter-clockwise order starting from the top left corner.

    Args:
        left (float): The x-coordinate of the left side of the rectangle.
        top (float): The y-coordinate of the top side of the rectangle.
        right (float): The x-coordinate of the right side of the rectangle.
        bottom (float): The y-coordinate of the bottom side of the rectangle.

    Returns:
        numpy.ndarray: A 2x2x4 array representing the rectangle coordinates. Each vertex is represented by an array
        of four values [x, y, tx, ty], where (x, y) are the vertex coordinates and (tx, ty) are texture coordinates.
    """
    return np.array([
        [[left, top, 0.0, 0.0], [right, top, 1.0, 0.0]],
        [[left, bottom, 0.0, 1.0], [right, bottom, 1.0, 1.0]]],
        dtype=np.float32)


def square(radius):
    """
    Creates a square shape centered at the origin with a specified radius. The square is represented similarly
    to the rectangle function, using a normalized coordinate system.

    Args:
        radius (float): The radius of the square, defined as half the length of a side. The square extends
        from -radius to +radius in both x and y directions.

    Returns:
        numpy.ndarray: A 2x2x4 array representing the square coordinates in the same format as the rectangle function.
    """
    return rectangle(-radius, -radius, radius, radius)


def fill_transform(slm, fit='short'):
    """
    Constructs a transformation matrix that makes a 'square' patch (range -1.0 to 1.0 in x and y) fill the SLM.
    The transformation depends on the 'fit' argument, which determines how the square patch is fitted onto the SLM.

    Args:
        slm: SLM object used to obtain the width and height dimensions.
        fit (str): The fitting strategy, which can be:
            - 'full': Makes the patch fill the full SLM, causing pixels in a square texture to become stretched.
            - 'short': Makes the patch square, filling the short side of the SLM.
            - 'small': Makes the patch square, filling the long side of the SLM. Part of the patch might be outside
              the SLM window if the SLM is not square.

    Returns:
        numpy.ndarray: A 3x3 transformation matrix used to adjust the square patch to fit the SLM as specified.

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
