import numpy as np
import geometry
from OpenGL.GL import *
from OpenGL.GL import shaders
from shaders import default_vertex_shader, default_fragment_shader, post_process_fragment_shader


class Patch:
    def __init__(self, slm, geometry, vertex_shader=default_vertex_shader, fragment_shader=default_fragment_shader):
        slm.patches.append(self)
        slm.activate()  # make sure the opengl operations occur in the context of the specified slm window
        (self._vertices, self._indices) = glGenBuffers(2)
        self.geometry = geometry

        # construct vertex shader, fragment shader and program
        vs = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self._program = shaders.compileProgram(vs, fs)
        self._phases = Texture(np.float32)

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self._vertices)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBindVertexBuffer(0, self._vertices, 0, 16)
        glUseProgram(self._program)

        glUniform1i(glGetUniformLocation(self._program, "texSampler"), 0)  # use texture unit 0 for the texture
        glActiveTexture(GL_TEXTURE0 + 0)
        glBindTexture(GL_TEXTURE_2D, self._phases.handle)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glDrawElements(GL_TRIANGLE_STRIP, self._index_count, GL_UNSIGNED_SHORT, None)

    @property
    def phases(self):
        """1-D or 2-D array holding phase values to display on the SLM.
        Phases are in radians, and stored as float32. There is no need to wrap the phase to a 0-2pi range.
        The values are only uploaded to the GPU when the setter is invoked (i.e. patch.phases = data).
        """
        return self._phases.data

    @phases.setter
    def phases(self, value):
        self._phases.set(value)

    @property
    def geometry(self):
        """Vertices that define the shape of the patch on the screen. Currently, this should be a NxMx4 numpy array
        of float32 values. Each 4 values define a vector: x,y position and tx, ty texture coordinate. See geometry.py
        for examples of geometry specifications. The vertices are drawn as a NxM 'rectangular' grid of quadrilaterals.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        # construct vertex buffer and index buffer
        # tell opengl how to interpret the data in the buffer (see vertex array object in SLM)
        # 0 = binding index for the vertex array object. 0 = offset into buffer. 16 = stride
        self._geometry = np.array(value, dtype=np.float32, copy=False)
        glBindBuffer(GL_ARRAY_BUFFER, self._vertices)
        glBufferData(GL_ARRAY_BUFFER, self._geometry.size * 4, self._geometry, GL_DYNAMIC_DRAW)

        # construct the indices that convert the vertices to a set of triangle strips (see triangle strip in OpenGL
        # specification)
        i = 0
        nr = self._geometry.shape[0]
        nc = self._geometry.shape[1]
        self._index_count = (nr - 1) * (2 * nc + 1)
        indices = np.zeros(self._index_count, np.uint16)
        for r in range(nr - 1):
            # emit triangle strip (single row)
            row_start = r * nc
            for c in range(nc):
                indices[i] = row_start + c
                indices[i + 1] = row_start + c + nc
                i += 2
            indices[i] = 0xFFFF
            i += 1
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._index_count * 2, indices, GL_DYNAMIC_DRAW)


class Texture:
    def __init__(self, dtype):
        self.dtype = dtype
        self.handle = glGenTextures(1)
        self.data = None

    def set(self, data):
        """ Note that textures holds a reference to the array that was last stored in it. A copy is only
            made when the array is not a numpy float32 array yet. If the data in the referenced array is modified,
            the data on the GPU and the data in 'phases' are out of sync. To synchronize them again, use
            patch.phases = data, or even patch.phases = patch.phases."""
        data = np.array(data, dtype=self.dtype, copy=False)
        reuse = self.data is not None and self.data.shape == data.shape  # reuse same texture memory if possible
        width = data.shape[0]
        height = 1 if data.ndim == 1 else data.shape[1]

        (internal_format, data_format) = (GL_R32F, GL_FLOAT) if self.dtype == np.float32 else (GL_R8, GL_BYTE)
        glBindTexture(GL_TEXTURE_2D, self.handle)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # alignment is at least 4 bytes since we are using float32 for everything
        if reuse:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, data_format, data)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RED, data_format, data)
        glBindTexture(GL_TEXTURE_2D, 0)  # unbind texture


class FrameBufferPatch(Patch):
    """Special patch that represents the frame buffer. All patches are first rendered to the frame buffer,
    and this buffer is rendered to the screen through a  final post-processing step that does the phase wrapping and
    implements the software lookup table."""

    def __init__(self, slm):
        super().__init__(slm, geometry.square(1.0), fragment_shader=post_process_fragment_shader)
        # Create a frame buffer object to render to. The frame buffer holds a texture that is the same size as the
        # window. All patches are first rendered to this texture. The texture
        # is then processed as a whole (applying the software lookup table) and displayed on the screen.
        glActiveTexture(GL_TEXTURE0)
        self._phases.set(np.zeros([slm.width, slm.height], dtype=np.float32))
        self._frame_buffer = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self._frame_buffer)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._phases.handle, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Could not construct frame buffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self._lookup_table = Texture(np.uint8)
        self.lookup_table = range(256)

    @property
    def lookup_table(self):
        """1-D array """
        return self._lookup_table.data

    @lookup_table.setter
    def lookup_table(self, value):
        self._lookup_table.set(value)
