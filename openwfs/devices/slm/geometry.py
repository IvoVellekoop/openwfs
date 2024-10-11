from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ...utilities import ExtentType, CoordinateType, unitless


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
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._indices = np.asarray(indices, dtype=np.uint16)

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


def rectangle(extent: ExtentType, center: CoordinateType = (0, 0)) -> Geometry:
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
    center = unitless(center)
    extent = unitless(extent)
    if extent.size == 1:
        extent = np.array((extent, extent))

    left = center[1] - 0.5 * extent[1]
    top = center[0] - 0.5 * extent[0]
    right = center[1] + 0.5 * extent[1]
    bottom = center[0] + 0.5 * extent[0]

    vertices = np.array(
        (
            [left, top, 0.0, 0.0],
            [right, top, 1.0, 0.0],
            [left, bottom, 0.0, 1.0],
            [right, bottom, 1.0, 1.0],
        ),
        dtype=np.float32,
    )

    indices = Geometry.compute_indices_for_grid((1, 1))
    return Geometry(vertices, indices)


def circular(
    radii: Sequence[float],
    segments_per_ring: Sequence[int],
    edge_count: int = 256,
    center: CoordinateType = (0, 0),
) -> Geometry:
    """Creates a circular geometry with the specified extent.

    This geometry maps a texture to a disk or a ring.
    It can be used for feedback-based wavefront shaping to match the segment size to the illumination profile of the
    SLM. [1]

    When used with a linear texture of shape `(1, np.sum(segments_per_ring)` (see `set_phases`),
    this geometry maps the pixels of the texture to concentric rings, starting with the innermost ring,
    and distributing the phase values counter-clockwise, starting at the positive x-axis.
    Each ring will hold the specified `segments_per_ring`.

    Notes:
         When `radii[0] == 0` , a disk is created.
         When `radii[0] > 0`, the center is left open, creating a ring

    Args:
        radii (Sequence[float]): The radii of the rings in the circular geometry.
            The first value is the inner radius of the inner ring, the last value is the outer radius of the outer ring.
        center (CoordinateType): The center of the circle with respect
            to the origin, specified in (y, x) coordinates.
            Default value is (0, 0).
        segments_per_ring (Sequence[int]): The number of segments for each ring.
            The number of segments is one less than the number of rings.
        edge_count (int): The number of edge points to approximate the full circle.
            The more edges, the closer the geometry will approximate a circle.
            Default value is 256.

    [1]: Mastiani, Bahareh, and Ivo M. Vellekoop. "Noise-tolerant wavefront shaping in a Hadamard basis." Optics
    express 29.11 (2021): 17534-17541.
    """
    ring_count = len(radii) - 1
    if len(segments_per_ring) != ring_count:
        raise ValueError(
            "The length of `radii` and `segments_per_ring` should both equal the number of rings (counting "
            "the inner disk as the first ring)."
        )

    # construct coordinates of points on a circle of radius 1.0
    # the start and end point coincide
    angles = np.linspace(0.0, 2 * np.pi, edge_count + 1)
    x = np.cos(angles)
    y = -np.sin(angles)
    x[0] = x[-1]  # make *exactly* equal
    y[0] = y[-1]

    # construct vertices.
    # each ring is a triangle strip with edge_count + 1 vertices on the inside and
    # edge_count + 1 vertices on the outside.
    vertices = np.zeros((ring_count, 2, edge_count + 1, 4), dtype=np.float32)
    x_inside = radii[0]  # coordinates of the vertices at the center of the ring
    y_inside = radii[0]
    segments_inside = 0
    total_segments = np.sum(segments_per_ring)
    for r in range(ring_count):
        x_outside = x * radii[r + 1]  # coordinates of the vertices at the outside of the ring
        y_outside = y * radii[r + 1]
        segments = segments_inside + segments_per_ring[r]
        vertices[r, 0, :, 0] = x_inside + center[1]
        vertices[r, 0, :, 1] = y_inside + center[0]
        vertices[r, 1, :, 0] = x_outside + center[1]
        vertices[r, 1, :, 1] = y_outside + center[0]
        vertices[r, :, :, 2] = (
            np.linspace(segments_inside, segments, edge_count + 1).reshape((1, -1)) / total_segments
        )  # tx
        x_inside = x_outside
        y_inside = y_outside
        segments_inside = segments

    vertices[:, 0, :, 3] = 0.0  # ty inside
    vertices[:, 1, :, 3] = 1.0  # ty outside

    # construct indices for a single ring, and repeat for all rings with the appropriate offset
    indices = Geometry.compute_indices_for_grid((1, edge_count)).reshape((1, -1))
    indices = indices + np.arange(ring_count).reshape((-1, 1)) * vertices.shape[1] * vertices.shape[2]
    indices[:, -1] = 0xFFFF
    return Geometry(vertices.reshape((-1, 4)), indices.reshape(-1))
