import numpy as np
import weakref
from OpenGL.GL import *
from OpenGL.GL import shaders
from .geometry import square
from .shaders import default_vertex_shader, default_fragment_shader, \
    post_process_fragment_shader, post_process_vertex_shader
from .texture import Texture


class Patch:
    PHASES_TEXTURE = 0  # indices of the phases texture in the _texture array

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

        slm.patches.append(self)
        slm.activate()  # make sure the opengl operations occur in the context of the specified slm window
        self.context = weakref.ref(slm)  # keep weak reference to parent, to avoid cyclic references
        self.geometry = geometry

        # construct vertex shader, fragment shader and program
        vs = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self._program = shaders.compileProgram(vs, fs)
        self._textures = [Texture(slm)]
        self.additive_blend = True
        self.enabled = True

    def draw(self):
        """Never call directly, this is called from slm.update()"""
        # glBindBuffer(GL_ARRAY_BUFFER, self._vertices) # not needed because we are binding the vertex buffer already?
        if not self.enabled:
            return

        glUseProgram(self._program)

        if self.additive_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)  # (1 * rgb, 1 * alpha)
            glBlendEquation(GL_FUNC_ADD)
        else:
            glDisable(GL_BLEND)

        for idx, texture in enumerate(self._textures):
            texture.synchronize()  # upload data again if it was modified
            glActiveTexture(GL_TEXTURE0 + idx)
            glBindTexture(texture.type, texture.handle)

        self._geometry.draw()

    @property
    def phases(self):
        """1-D or 2-D array holding phase values to display on the SLM.
        Phases are in radians, and stored as float32. There is no need to wrap the phase to a 0-2pi range.
        The values are uploaded to the GPU automatically on slm.update.
        """
        return self._textures[Patch.PHASES_TEXTURE].data

    @phases.setter
    def phases(self, value):
        self._textures[Patch.PHASES_TEXTURE].data = value

    @property
    def geometry(self):
        """Vertices that define the shape of the patch on the screen. Currently, this should be a NxMx4 numpy array
        of float32 values. Each 4 values define a vector: x,y position and tx, ty texture coordinate. See geometry.py
        for examples of geometry specifications. The vertices are drawn as a NxM 'rectangular' grid of quadrilaterals.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            value = Geometry(self.context(), value)
        self._geometry = value


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


class FrameBufferPatch(Patch):
    """Special patch that represents the frame buffer. All patches are first rendered to the frame buffer,
    and this buffer is rendered to the screen through a  final post-processing step that does the phase wrapping and
    implements the software lookup table."""

    LUT_TEXTURE = 1

    def __init__(self, slm):
        super().__init__(slm, square(1.0), fragment_shader=post_process_fragment_shader,
                         vertex_shader=post_process_vertex_shader)
        # Create a frame buffer object to render to. The frame buffer holds a texture that is the same size as the
        # window. All patches are first rendered to this texture. The texture
        # is then processed as a whole (applying the software lookup table) and displayed on the screen.
        self.frame_buffer = glGenFramebuffers(1)

        self.phases = np.zeros([slm.width, slm.height], dtype=np.float32)
        self._textures[Patch.PHASES_TEXTURE].synchronize()
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               self._textures[Patch.PHASES_TEXTURE].handle, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Could not construct frame buffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self._textures.append(Texture(slm, GL_TEXTURE_1D))  # create texture for lookup table
        self.lookup_table = np.arange(0, 256) / 255.0 # 256 entries from 0.0 to 1.0
        self.additive_blend = False

    def __del__(self):
        if self.context() is not None and hasattr(self, 'frame_buffer'):
            self.context().activate()
            glDeleteFramebuffers(1, [self.frame_buffer])

    @property
    def lookup_table(self):
        """1-D array """
        return self._textures[FrameBufferPatch.LUT_TEXTURE].data

    @lookup_table.setter
    def lookup_table(self, value):
        self.context().activate()
        self._textures[FrameBufferPatch.LUT_TEXTURE].data = value


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
        glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0)  # first two float32 are screen coordinates
        glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8)  # second two are texture coordinates
        glVertexAttribBinding(0, 0)  # use binding index 0 for both attributes
        glVertexAttribBinding(1, 0)  # the attribute format can now be used with glBindVertexBuffer

        # enable primitive restart, so that we can draw multiple triangle strips with a single draw call
        glEnable(GL_PRIMITIVE_RESTART)
        glPrimitiveRestartIndex(0xFFFF)  # this is the index we use to separate individual triangle strips
