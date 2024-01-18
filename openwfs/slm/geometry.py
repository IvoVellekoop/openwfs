import numpy as np
import weakref
from OpenGL.GL import *


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


class Geometry:
    def __init__(self, context, vertices):
        self.context = weakref.ref(context)  # keep weak reference to parent, to avoid cyclic references

        if vertices is None:
            self.vertices = square(1.0)
            self.indices = Geometry.compute_indices_for_grid(self.vertices.shape)
        elif isinstance(vertices, tuple):
            self.vertices = np.array(vertices[0], dtype=np.float32, copy=False)
            self.indices = np.array(vertices[1], dtype=np.uint16, copy=False)
        else:
            self.vertices = np.array(vertices, dtype=np.float32, copy=False)
            self.indices = Geometry.compute_indices_for_grid(self.vertices.shape)

        # store the data on the GPU
        self.context().activate()
        (self._vertices, self._indices) = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, self._vertices)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.size * 4, self.vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.size * 2, self.indices, GL_DYNAMIC_DRAW)

    def __del__(self):
        if self.context() is not None and hasattr(self, '_vertices'):
            self.context().activate()
            glDeleteBuffers(2, [self._vertices, self._indices])

    @staticmethod
    def compute_indices_for_grid(shape):
        # construct the indices that convert the vertices to a set of triangle strips (see triangle strip in OpenGL
        # specification)
        assert len(shape) == 3  # expect 2-D array of vertices
        assert shape[2] == 4  # where each vertex holds 4 floats
        i = 0
        nr = shape[0]
        nc = shape[1]
        index_count = (nr - 1) * (2 * nc + 1)
        indices = np.zeros(index_count, np.uint16)
        for r in range(nr - 1):
            # emit triangle strip (single row)
            row_start = r * nc
            for c in range(nc):
                indices[i] = row_start + c
                indices[i + 1] = row_start + c + nc
                i += 2
            indices[i] = 0xFFFF
            i += 1
        return indices

    def draw(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBindVertexBuffer(0, self._vertices, 0, 16)
        glDrawElements(GL_TRIANGLE_STRIP, self.indices.size, GL_UNSIGNED_SHORT, None)
