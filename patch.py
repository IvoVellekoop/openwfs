import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GL import arrays
from shaders import default_vertex_shader, default_fragment_shader


class Patch:
    def __init__(self, slm, geometry, vertex_shader=default_vertex_shader, fragment_shader=default_fragment_shader):
        slm.patches.append(self)
        slm.activate() # make sure the opengl operations occur in the context of the specified slm window
        (self._vertices, self._indices) = glGenBuffers(2)
        self.geometry = geometry

        # construct vertex shader, fragment shader and program
        vs = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)
        self.texture = glGenTextures(1)
        self._phases = None

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self._vertices)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indices)
        glBindVertexBuffer(0, self._vertices, 0, 16)
        glUseProgram(self.program)

        glUniform1i(glGetUniformLocation(self.program, "texSampler"), 0)  # use texture unit 0 for the texture
        glActiveTexture(GL_TEXTURE0 + 0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
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
        return self._phases

    @phases.setter
    def phases(self, value):
        self._phases = set_texture(self._phases, value, np.float32, self.texture)

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


def set_texture(original, data, dtype, texture_id):
    """ Note that textures holds a reference to the array that was last stored in it. A copy is only
        made when the array is not a numpy float32 array yet. If the data in the referenced array is modified,
        the data on the GPU and the data in 'phases' are out of sync. To synchronize them again, use
        patch.phases = data, or even patch.phases = patch.phases."""
    data = np.array(data, dtype=dtype, copy=False)
    reuse = original is not None and original.shape == data.shape  # reuse same texture memory if possible
    width = data.shape[0]
    height = data.shape[1]

    (internal_format, format) = (GL_R32F, GL_FLOAT) if dtype == np.float32 else (GL_R8, GL_BYTE)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # alignment is at least 4 bytes since we are using float32 for everything
    if reuse:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, format, data)
    else:
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RED, format, data)
    glBindTexture(GL_TEXTURE_2D, 0)  # unbind texture
    return data
