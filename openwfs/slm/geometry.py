import numpy as np
from typing import Optional
from ..utilities import ExtentType, CoordinateType, unitless
from numpy.typing import ArrayLike


class Geometry:
    """Class that represents the shape of a Patch object that can be drawn on the screen.

    The geometry is defined by a set of vertices
    and a set of indices that define the order in which the vertices are drawn.

    The vertices are stored in a numpy array of shape (N, 4) where N is the number of vertices.
    Each vertex is represented by an array of four values [x, y, tx, ty].
    Here, (x, y) are the coordinates that determine where the vertex is drawn on the screen (after the
    applying the `transform` of the SLM that contains this patch).

    The vertices and indices define a triangle strip (see OpenGL specification) that is drawn on the screen.
    A triangle strip is a sequence of connected triangles, where each triangle shares two vertices with the previous
    triangle. The indices are used to determine the order in which the vertices are connected to form the triangles.
    To start a new triangle strip, insert the special index 0xFFFF into the index array.

    (tx, ty) are the texture coordinates that determine which pixel of the texture
    (e.g. the array passed to `set_phases`) is drawn at each vertex.
    For each triangle, the screen coordinates (x,y) define a triangle on the screen, whereas the texture coordinates
    (tx, ty) define a triangle in the texture.
    OpenGL maps the texture triangle onto the screen triangle, using linear interpolation of the coordinates between
    the vertex points.

    The vertex data is uploaded to the GPU when the Geometry object is assigned to the `geometry` property of a Patch.
    """

    def __init__(self, vertices: ArrayLike, indices: ArrayLike):
        """
        Constructs a new Geometry object.

        Args:
            vertices (np.ndarray): The vertices of the geometry. See class documentation for details.
            indices (np.ndarray): The indices of the geometry. See class documentation for details.
        """
        self._vertices = np.array(vertices, dtype=np.float32, copy=False)
        self._indices = np.array(indices, dtype=np.uint16, copy=False)

        if self._vertices.ndim != 2 or self._vertices.shape[1] != 4:
            raise ValueError("Vertices should be a 2-D array with 4 columns")
        if self._indices.ndim != 1:
            raise ValueError("Indices should be a 1-D array")

    @property
    def vertices(self):
        """The vertices of the geometry."""
        return self._vertices

    @property
    def indices(self):
        """The indices of the geometry."""
        return self._indices

    @staticmethod
    def compute_indices_for_grid(shape):
        """
        Computes indices for a rectangular grid of vertices.

        Assuming the vertices represent the corner points of a rectangular grid,
        this function computes the indices needed to connect these vertices into triangle strips.

        Args:
            shape (Sequence[int]): The shape of the grid (nr, nc).
                nr is the number of grid rows, nc is the number of grid columns.
                Note that the number of vertices should equal (nr + 1) · (nc+1).
        """
        i = 0
        nr = shape[0]
        nc = shape[1] + 1
        index_count = nr * (2 * nc + 1)
        indices = np.zeros(index_count, np.uint16)
        for r in range(nr):
            # emit triangle strip (single row)
            row_start = r * nc
            for c in range(nc):
                indices[i] = row_start + c
                indices[i + 1] = row_start + c + nc
                i += 2
            indices[i] = 0xFFFF
            i += 1
        return indices


def rectangle(extent: ExtentType, center: Optional[CoordinateType] = None) -> Geometry:
    """
    Creates a rectangle geometry with the specified extent.

    Args:
        extent (ExtentType): The extent (height, width) of the rectangle.
            Default value is (2, 2).
            If a single value is specified, the same value is used for both axes.
        center (CoordinateType): The center of the rectangle with respect
            to the origin, specified in (y, x) coordinates.
            Default value is (0, 0).
    """
    if center is None:
        center = np.array((0.0, 0.0))
    else:
        center = unitless(center)
    extent = unitless(extent)
    if extent.size == 1:
        extent = np.array((extent, extent))

    left = center[1] - 0.5 * extent[1]
    top = center[0] - 0.5 * extent[0]
    right = center[1] + 0.5 * extent[1]
    bottom = center[0] + 0.5 * extent[0]

    vertices = np.array(([left, top, 0.0, 0.0], [right, top, 1.0, 0.0],
                         [left, bottom, 0.0, 1.0], [right, bottom, 1.0, 1.0]), dtype=np.float32)

    indices = Geometry.compute_indices_for_grid((1, 1))
    return Geometry(vertices, indices)
