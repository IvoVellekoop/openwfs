import numpy as np
from geometry import square
import weakref
from OpenGL.GL import *
from OpenGL.GL import shaders
from shaders import default_vertex_shader, default_fragment_shader, \
    post_process_fragment_shader, post_process_vertex_shader


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
            glBlendFunc(GL_ONE, GL_ONE) # (1 * rgb, 1 * alpha)
            glBlendEquation(GL_FUNC_ADD)
        else:
            glDisable(GL_BLEND)

        for idx, texture in enumerate(self._textures):
            glActiveTexture(GL_TEXTURE0 + idx)
            glBindTexture(texture.type, texture.handle)

        self._geometry.draw()

    @property
    def phases(self):
        """1-D or 2-D array holding phase values to display on the SLM.
        Phases are in radians, and stored as float32. There is no need to wrap the phase to a 0-2pi range.
        The values are only uploaded to the GPU when the setter is invoked (i.e. patch.phases = data).
        """
        return self._textures[Patch.PHASES_TEXTURE].data

    @phases.setter
    def phases(self, value):
        self.context().activate()  # activate OpenGL context for this SLM (needed in case we have multiple SLMs)
        self._textures[Patch.PHASES_TEXTURE].set(value)

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
            vertices = square(1.0)
        if not isinstance(vertices, tuple):
            vertices = (vertices, Geometry.compute_indices_for_grid(vertices.shape))

        # store the data on the GPU
        self.context().activate()
        (self._vertices, self._indices) = glGenBuffers(2)
        self.vertices = np.array(vertices[0], dtype=np.float32, copy=False)
        self.indices = np.array(vertices[1], dtype=np.uint16, copy=False)
        glBindBuffer(GL_ARRAY_BUFFER, self._vertices)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.size * 4, self.vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.size * 2, self.indices, GL_DYNAMIC_DRAW)

    def __del__(self):
        if self.context() is not None:
            self.context().activate()
            glDeleteBuffers(2, [self._vertices, self._indices])

    @staticmethod
    def compute_indices_for_grid(shape):
        # construct the indices that convert the vertices to a set of triangle strips (see triangle strip in OpenGL
        # specification)
        assert len(shape) == 3  # expect 2-D array of vertices
        assert shape[2] == 4    # where each vertex holds 4 floats
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


class Texture:
    def __init__(self, context, texture_type=GL_TEXTURE_2D):
        self.context = weakref.ref(context)
        self.handle = glGenTextures(1)
        self.type = texture_type
        glBindTexture(self.type, self.handle)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_R, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(self.type, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        self.data = None

    def __del__(self):
        if self.context() is not None:
            self.context().activate()
            glDeleteTextures(1, [self.handle])

    def set(self, data):
        """ Set texture data. The dimensionality of the data should match that of the texture.
            As an exception, all texture types can be set to a scalar value.
            Note that textures holds a reference to the array that was last stored in it. A copy is only
            made when the array is not a numpy float32 array yet. If the data in the referenced array is modified,
            the data on the GPU and the data in 'phases' are out of sync. To synchronize them again, use
            patch.phases = data, or even patch.phases = patch.phases."""
        self.context().activate()
        data = np.array(data, dtype=np.float32, copy=False)
        reuse = self.data is not None and self.data.shape == data.shape  # reuse same texture memory if possible
        (internal_format, data_format, data_type) = (GL_R32F, GL_RED, GL_FLOAT)
        glBindTexture(self.type, self.handle)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # alignment is at least 4 bytes since we are using float32 for everything

        if self.type == GL_TEXTURE_1D:
            if data.ndim == 0:
                data.shape = 1
            elif data.ndim != 1:
                raise ValueError("Data should be 1-d array")
            if reuse:
                glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 0, data.shape[0], data_format, data_type, data)
            else:
                glTexImage1D(GL_TEXTURE_1D, 0, internal_format, data.shape[0], 0, data_format, data_type, data)

        elif self.type == GL_TEXTURE_2D:
            if data.ndim == 0:
                data.shape = (1, 1)
            elif data.ndim != 2:
                raise ValueError("Data should be 2-d array")
            if reuse:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data.shape[0], data.shape[1], data_format, data_type, data)
            else:
                glTexImage2D(GL_TEXTURE_2D, 0, internal_format, data.shape[0], data.shape[1], 0, data_format, data_type,
                             data)

        else:
            raise ValueError("Texture type not supported")

        self.data = data    # store data so that it can be read back by users


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
        self.phases = np.zeros([slm.width, slm.height], dtype=np.float32)
        self.frame_buffer = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               self._textures[Patch.PHASES_TEXTURE].handle, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Could not construct frame buffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self._textures.append(Texture(slm, GL_TEXTURE_1D))  # create texture for lookup table
        self.lookup_table = np.arange(0.0, 1.0, 1/255.0)
        self.additive_blend = False

    def __del__(self):
        if self.context() is not None:
            self.context().activate()
            glDeleteFramebuffers(1, self.frame_buffer)

    @property
    def lookup_table(self):
        """1-D array """
        return self._textures[FrameBufferPatch.LUT_TEXTURE].data

    @lookup_table.setter
    def lookup_table(self, value):
        self.context().activate()
        self._textures[FrameBufferPatch.LUT_TEXTURE].set(value)
