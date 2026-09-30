import OpenGL.GL as GL
import numpy as np

from .context import Context
from OpenGL.arrays.numpymodule import GL_TYPE_TO_ARRAY_MAPPING


class Texture:
    def __init__(
        self, slm, texture_type=None, internal_format=GL.GL_R32F, format_type=GL.GL_RED, data_type=GL.GL_FLOAT
    ):
        self.context = Context(slm)
        self.handle = GL.glGenTextures(1)
        self.type = texture_type if texture_type is not None else GL.GL_TEXTURE_2D
        self.synchronized = False  # self.data is not yet synchronized with texture in GPU memory
        self._data_shape = None  # current size of the texture, to see if we need to make a new texture or

        self._internal_format = internal_format
        self._format_type = format_type
        self._data_type = data_type

        # overwrite the exiting one

        # create a single pixel texture as default (also activates the OpenGL context and binds the texture
        self.set_data(0)

        # set wrapping and interpolation options
        GL.glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(self.type, GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT)
        GL.glTexParameteri(self.type, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.type, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

    def __del__(self):
        with self.context as slm:
            if slm:
                GL.glDeleteTextures(1, [self.handle])

    def _bind(self, idx):
        """Bind texture to texture unit idx. Assumes that the OpenGL context is already active."""
        GL.glActiveTexture(GL.GL_TEXTURE0 + idx)
        GL.glBindTexture(self.type, self.handle)

    def set_data(self, value):
        """Set texture data.

        The texture data is directly copied to the GPU memory,
         so the original data array can be modified or deleted.
        """

        with self.context:
            GL.glBindTexture(self.type, self.handle)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)  # alignment is at least four bytes since we use float32

            value = np.asarray(value, dtype=GL_TYPE_TO_ARRAY_MAPPING[self._data_type], order="C")

            if self.type == GL.GL_TEXTURE_1D:
                # check if data has the correct dimension, convert scalars to arrays of correct dimension
                if value.ndim == 0:
                    value = value.reshape((1,))
                elif value.ndim != 1:
                    raise ValueError("Data should be a 1-d array or a scalar")

                if value.shape != self._data_shape:
                    # create a new texture
                    GL.glTexImage1D(
                        GL.GL_TEXTURE_1D,
                        0,
                        self._internal_format,
                        value.shape[0],
                        0,
                        self._format_type,
                        self._data_type,
                        value,
                    )
                    self._data_shape = value.shape
                else:
                    # overwrite existing texture
                    GL.glTexSubImage1D(
                        GL.GL_TEXTURE_1D,
                        0,
                        0,
                        value.shape[0],
                        self._format_type,
                        self._data_type,
                        value,
                    )

            elif self.type == GL.GL_TEXTURE_2D:
                if value.ndim == 0:
                    value = value.reshape((1, 1))
                elif value.ndim != 2:
                    raise ValueError("Data should be a 2-D array or a scalar")

                if value.shape != self._data_shape:
                    GL.glTexImage2D(
                        GL.GL_TEXTURE_2D,
                        0,
                        self._internal_format,
                        value.shape[1],
                        value.shape[0],
                        0,
                        self._format_type,
                        self._data_type,
                        value,
                    )
                    self._data_shape = value.shape
                else:
                    GL.glTexSubImage2D(
                        GL.GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        value.shape[1],
                        value.shape[0],
                        self._format_type,
                        self._data_type,
                        value,
                    )
            else:
                raise ValueError("Texture type not supported")

    def get_data(self):
        with self.context:
            if self._format_type == GL.GL_RED:
                shape = self._data_shape + (1,)
            else:  # self._format_type == GL.GL_RGB
                shape = self._data_shape + (3,)

            data = np.empty(shape, dtype=GL_TYPE_TO_ARRAY_MAPPING[self._data_type])
            GL.glGetTextureImage(self.handle, 0, self._format_type, self._data_type, data.size * 4, data)
            return data
