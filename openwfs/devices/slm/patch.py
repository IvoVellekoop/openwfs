import warnings
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .context import Context

try:
    import OpenGL.GL as GL
    from OpenGL.GL import glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers, glEnable, glBlendFunc, \
        glBlendEquation, glDisable, glUseProgram, glBindVertexBuffer, glDrawElements, glGenFramebuffers, \
        glBindFramebuffer, glFramebufferTexture2D, glCheckFramebufferStatus, glDeleteFramebuffers, \
        glEnableVertexAttribArray, glVertexAttribFormat, glVertexAttribBinding, glEnableVertexAttribArray, \
        glPrimitiveRestartIndex, glActiveTexture, glBindTexture, glGenVertexArrays, glBindVertexArray

    from OpenGL.GL import shaders
except AttributeError:
    warnings.warn("OpenGL not found, SLM will not work")
from .geometry import rectangle, Geometry
from .shaders import default_vertex_shader, default_fragment_shader, \
    post_process_fragment_shader, post_process_vertex_shader
from .texture import Texture
from ...core import PhaseSLM


class Patch(PhaseSLM):
    _PHASES_TEXTURE = 0  # indices of the phases texture in the _texture array

    def __init__(self, slm, geometry=None, vertex_shader=default_vertex_shader,
                 fragment_shader=default_fragment_shader):
        """
        Constructs a new patch (a shape) that can be drawn on the screen.
        By default, the patch is a square with 'radius' 1.0 (width and height 2.0) centered at 0.0, 0.0
        To specify a different geometry, provide either:
          - a 2-D array of vertices, such as produced by the 'geometry' module. The vertices are interpreted as points
            on a (possibly deformed) grid.
          - a tuple containing a 1-D array of vertices and a 1-D array of indices. The indices indicate how the
            vertices are connected into triangles (see Geometry object for details).
          - an existing Geometry object. It is possible to attach the same Geometry object to multiple patches.
            Note, however, that Geometry objects cannot be shared between different SLMs.
        """
        self._vertices = None
        self._indices = None
        self._index_count = 0
        self.additive_blend = True
        self.enabled = True
        self.context = Context(slm)

        # construct vertex shader, fragment shader and program
        with self.context:
            vs = shaders.compileShader(vertex_shader, GL.GL_VERTEX_SHADER)
            fs = shaders.compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER)
            self._program = shaders.compileProgram(vs, fs)
            self._textures = [Texture(self.context)]

        self.geometry = rectangle(2.0) if geometry is None else geometry
        super().__init__()

    def __del__(self):
        self._delete_buffers()

    def _draw(self):
        """Never call directly, this is called from slm.update()"""
        if not self.enabled:
            return

        glUseProgram(self._program)

        if self.additive_blend:
            glEnable(GL.GL_BLEND)
            glBlendFunc(GL.GL_ONE, GL.GL_ONE)  # (1 * rgb, 1 * alpha)
            glBlendEquation(GL.GL_FUNC_ADD)
        else:
            glDisable(GL.GL_BLEND)

        for idx, texture in enumerate(self._textures):
            # activate texture as texture unit idx
            texture._bind(idx)  # noqa: ok to use _bind in friend class

        # perform the actual drawing
        glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBindVertexBuffer(0, self._vertices, 0, 16)
        glDrawElements(GL.GL_TRIANGLE_STRIP, self._index_count, GL.GL_UNSIGNED_SHORT, None)

    def set_phases(self, values: ArrayLike, update=True):
        """
        Args:
            values(ArrayLike): 1-D or 2-D array holding phase values to display on the SLM.
                Phases are in radians, and stored as float32. There is no need to wrap the phase to a 0-2pi range.
            update(bool): when True, the SLM in which this patch is contained is updated immediately.
                When False, the SLM is not updated, and the
                caller is responsible for calling slm.update() to update the SLM.
        """
        self._textures[Patch._PHASES_TEXTURE].set_data(values)
        if update:
            self.update()

    def update(self):
        self.context.slm.update()

    def _delete_buffers(self):
        with self.context as slm:
            if slm:
                glDeleteBuffers(2, [self._vertices, self._indices])

    @property
    def geometry(self):
        """Vertices that define the shape of the patch on the screen. Currently, this should be a NxMx4 numpy array
        of float32 values. Each 4 values define a vector: x,y position and tx, ty texture coordinate. See geometry.py
        for examples of geometry specifications. The vertices are drawn as a NxM 'rectangular' grid of quadrilaterals.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value: Geometry):
        if not isinstance(value, Geometry):
            raise ValueError("Geometry should be a Geometry object")

        # store the data on the GPU
        with self.context:
            self._geometry = value
            (self._vertices, self._indices) = glGenBuffers(2)
            self._index_count = value.indices.size
            glBindBuffer(GL.GL_ARRAY_BUFFER, self._vertices)
            glBufferData(GL.GL_ARRAY_BUFFER, value.vertices.size * 4, value.vertices, GL.GL_DYNAMIC_DRAW)
            glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indices)
            glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, value.indices.size * 2, value.indices, GL.GL_DYNAMIC_DRAW)


class FrameBufferPatch(Patch):
    """Special patch that represents the frame buffer. All patches are first rendered to the frame buffer,
    and this buffer is rendered to the screen through a  final post-processing step that does the phase wrapping and
    implements the software lookup table."""

    _LUT_TEXTURE = 1
    _textures: list[Texture]

    def __init__(self, slm, lookup_table: Sequence[int]):
        super().__init__(slm, fragment_shader=post_process_fragment_shader,
                         vertex_shader=post_process_vertex_shader)
        # Create a frame buffer object to render to. The frame buffer holds a texture that is the same size as the
        # window. All patches are first rendered to this texture. The texture
        # is then processed as a whole (applying the software lookup table) and displayed on the screen.
        self._frame_buffer = glGenFramebuffers(1)

        self.set_phases(np.zeros(self.context.slm.shape, dtype=np.float32), update=False)
        glBindFramebuffer(GL.GL_FRAMEBUFFER, self._frame_buffer)
        glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D,
                               self._textures[Patch._PHASES_TEXTURE].handle, 0)
        if glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Could not construct frame buffer")
        glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self._textures.append(Texture(self.context, GL.GL_TEXTURE_1D))  # create texture for lookup table
        self._lookup_table = None
        self.lookup_table = lookup_table
        self.additive_blend = False

    def __del__(self):
        with self.context as slm:
            if slm:
                glDeleteFramebuffers(1, [self._frame_buffer])

    @property
    def lookup_table(self):
        """1-D array """
        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, value):
        self._lookup_table = np.array(value)
        self._textures[FrameBufferPatch._LUT_TEXTURE].set_data(self._lookup_table / 255)

    def get_pixels(self):
        data = self._textures[FrameBufferPatch._PHASES_TEXTURE].get_data()

        # flip data upside down, because the OpenGL convention is to have the origin at the bottom left,
        # but we want it at the top left (like in numpy)
        return data[::-1, :]


class VertexArray:
    # A VertexArray informs OpenGL about the format of the vertex data we will use.
    # Each vertex contains four float32 components:
    # x, y coordinate for vertex position. Will be transformed by the transform matrix.
    # tx, ty texture coordinates, range from 0.0, 0.0 to 1.0, 1.0 to cover the full texture
    #
    # To inform OpenGL about this format, we create vertex an array object and store format properties.
    # The elements of the vertex are available to a vertex shader as vec2 position (location 0)
    # and vec2 texture_coordinates (location 1), see vertex shader in Patch for an example.
    # All this information is bound to a binding index before use by calling glBindVertexBuffer,
    # which is done when a vertex buffer is created (see Patch).
    #
    # Since we have a fixed vertex format, we only need to bind the VertexArray once, and not bother with
    # updating, binding, or even deleting it
    def __init__(self):
        self._vertex_array = glGenVertexArrays(1)  # no need to destroy explicitly, destroyed when window is destroyed
        glBindVertexArray(self._vertex_array)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribFormat(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0)  # first two float32 are screen coordinates
        glVertexAttribFormat(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 8)  # second two are texture coordinates
        glVertexAttribBinding(0, 0)  # use binding index 0 for both attributes
        glVertexAttribBinding(1, 0)  # the attribute format can now be used with glBindVertexBuffer

        # enable primitive restart, so that we can draw multiple triangle strips with a single draw call
        glEnable(GL.GL_PRIMITIVE_RESTART)
        glPrimitiveRestartIndex(0xFFFF)  # this is the index we use to separate individual triangle strips
