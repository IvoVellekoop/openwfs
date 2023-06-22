import numpy as np
from math import pi
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GL import arrays

class Patch:
    def __init__(self, vertex_grid):
        # construct vertex buffer and index buffer
        vertex_grid = np.array(vertex_grid, dtype=np.float32)
        self.vertices = arrays.vbo.VBO(vertex_grid)
        i = 0
        nr = vertex_grid.shape[0]
        nc = vertex_grid.shape[1]
        indices = np.zeros((nr - 1) * (2 * nc + 1), np.uint16)
        for r in range(nr - 1):
            # emit triangle strip (single row)
            row_start = r * nc
            for c in range(nc):
                indices[i] = row_start + c
                indices[i + 1] = row_start + c + nc
                i += 2
            indices[i] = 0xFFFF
            i += 1

        self.indices = arrays.vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

        # tell opengl how to interpret the data in the buffer (see vertex array object in SLM)
        # 0 = binding index for the vertex array object. 0 = offset into buffer. 16 = stride
        glBindVertexBuffer(0, self.vertices, 0, 16)

        # construct vertex shader, fragment shader and program
        vertex_shader_code = """
            #version 440 core
            layout(location = 0) in vec2 slm_coordinate;
            layout(location = 1) in vec2 texture_coordinate;
            layout(binding = 1, std140) uniform globals {
                mat3x2 transform;
            };
            out vec2 texCoord;
            
            void main() {
                vec2 pos = transform * vec3(slm_coordinate, 1.0);
                gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
                texCoord = texture_coordinate;
            }
            """

        fragment_shader_code = """
            #version 440 core
            in vec2 texCoord;
            out vec4 colorOut;
            uniform sampler2D texSampler;
            const float scale = 0.1591549431f; // = 1 / 2pi
            
            void main() {
                float val = texture(texSampler, texCoord).r * scale;
                colorOut = vec4(val, val, val, 1.0);
            }
            """

        vs = shaders.compileShader(vertex_shader_code, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)

        # set up texture
        rng = np.random.default_rng()
        data = rng.random([10, 10], np.float32) * 2.0 * pi
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # alignment is at least 4 bytes since we are using float32 for everything
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, data.shape[0], data.shape[1], 0, GL_RED, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_2D, 0)  # unbind texture

    def draw(self):
        self.vertices.bind()
        self.indices.bind()
        glUseProgram(self.program)

        glUniform1i(glGetUniformLocation(self.program, "texSampler"), 0)  # use texture unit 0 for the texture
        glActiveTexture(GL_TEXTURE0 + 0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glDrawElements(GL_TRIANGLE_STRIP, self.indices.data.size, GL_UNSIGNED_SHORT, self.indices)
        pass
