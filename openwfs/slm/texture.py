import numpy as np
import warnings

from .context import Context

try:
    import OpenGL.GL as GL
    from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexSubImage2D, glTexImage1D, glTexSubImage1D, \
        glTexParameteri, glActiveTexture, glDeleteTextures, glGetTextureImage, glPixelStorei
except AttributeError:
    warnings.warn("OpenGL not found, SLM will not work"),


class Texture:
    def __init__(self, slm, texture_type=GL.GL_TEXTURE_2D):
        self.context = Context(slm)
        self.handle = glGenTextures(1)
        self.type = texture_type
        self.synchronized = False  # self.data is not yet synchronized with texture in GPU memory
        self._data_shape = None  # current size of the texture, to see if we need to make a new texture or
        # overwrite the exiting one

        # create a single pixel texture as default (also activates the OpenGL context and binds the texture
        self.set_data(0)

        # set wrapping and interpolation options
        glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT)
        glTexParameteri(self.type, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        glTexParameteri(self.type, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

    def __del__(self):
        with self.context as slm:
            if slm:
                glDeleteTextures(1, [self.handle])

    def _bind(self, idx):
        """ Bind texture to texture unit idx. Assumes that the OpenGL context is already active."""
        glActiveTexture(GL.GL_TEXTURE0 + idx)
        glBindTexture(self.type, self.handle)

    def set_data(self, value):
        """ Set texture data.

        The texture data is directly copied to the GPU memory,
         so the original data array can be modified or deleted.
        """
        value = np.array(value, dtype=np.float32, order='C', copy=False)

        with self.context:
            glBindTexture(self.type, self.handle)
            glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)  # alignment is at least four bytes since we use float32
            (internal_format, data_format, data_type) = (GL.GL_R32F, GL.GL_RED, GL.GL_FLOAT)

            if self.type == GL.GL_TEXTURE_1D:
                # check if data has the correct dimension, convert scalars to arrays of correct dimension
                if value.ndim == 0:
                    value = value.reshape((1,))
                elif value.ndim != 1:
                    raise ValueError("Data should be a 1-d array or a scalar")
                if value.shape != self._data_shape:
                    # create a new texture
                    glTexImage1D(GL.GL_TEXTURE_1D, 0, internal_format, value.shape[0], 0, data_format, data_type, value)
                    self._data_shape = value.shape
                else:
                    # overwrite existing texture
                    glTexSubImage1D(GL.GL_TEXTURE_1D, 0, 0, value.shape[0], data_format, data_type, value)

            elif self.type == GL.GL_TEXTURE_2D:
                if value.ndim == 0:
                    value = value.reshape((1, 1))
                elif value.ndim != 2:
                    raise ValueError("Data should be a 2-d array or a scalar")
                if value.shape != self._data_shape:
                    glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, value.shape[1], value.shape[0], 0,
                                 data_format, data_type, value)
                    self._data_shape = value.shape
                else:
                    glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, value.shape[1], value.shape[0], data_format,
                                    data_type, value)
            else:
                raise ValueError("Texture type not supported")

    def get_data(self):
        with self.context:
            data = np.empty(self._data_shape, dtype='float32')
            glGetTextureImage(self.handle, 0, GL.GL_RED, GL.GL_FLOAT, data.size * 4, data)
            return data
