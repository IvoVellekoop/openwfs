import OpenGL.GL as GL
import numpy as np

from .context import Context


class Texture:
    def __init__(self, slm, texture_type=None, encoding="8b_r"):
        self.context = Context(slm)
        self.handle = GL.glGenTextures(1)
        self.type = texture_type if texture_type is not None else GL.GL_TEXTURE_2D
        self.synchronized = False  # self.data is not yet synchronized with texture in GPU memory
        self._data_shape = None  # current size of the texture, to see if we need to make a new texture or
        # overwrite the exiting one

        self.encoding = encoding

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
        if self.encoding == "8b_r":
            self.set_data_8b_r(value)
        elif self.encoding == "10b_rb":
            self.set_data_10b_rb(value)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def set_data_8b_r(self, value):
        """Set texture data.

        The texture data is directly copied to the GPU memory,
         so the original data array can be modified or deleted.
        """
        value = np.asarray(value, dtype=np.float32, order="C")

        with self.context:
            GL.glBindTexture(self.type, self.handle)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)  # alignment is at least four bytes since we use float32
            internal_format, data_format, data_type = (
                GL.GL_R32F,
                GL.GL_RED,
                GL.GL_FLOAT,
            )

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
                        internal_format,
                        value.shape[0],
                        0,
                        data_format,
                        data_type,
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
                        data_format,
                        data_type,
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
                        internal_format,
                        value.shape[1],
                        value.shape[0],
                        0,
                        data_format,
                        data_type,
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
                        data_format,
                        data_type,
                        value,
                    )
            else:
                raise ValueError("Texture type not supported")

    @staticmethod
    def convert_floatdata_to_10b_rb(data):
        data = np.asarray(data, dtype=np.float32, order="C")
        # Creates a 32-bit RGBA image from a 2D array of float values in the range [0, 1]. In this image, only the 10 least significant bits have information. The bits need to be moved to the right bits of the RGBA channels.
        value = np.rint(data * 1023).astype(np.uint32)

        # Calculates the 2 least significant bits of the img which need to be moved to the 2 least significant bits of the blue channel
        least_significant_bits = value & 0b11

        # Move the bits to be filling the 2 least significant bits of the blue channel
        least_significant_bits = least_significant_bits << 8

        # Remove the 2 least significant bits of the image which will be added later to the right channel
        value = value >> 2

        # Move the bits to be filling the 8 bits of the red channel
        value = value << 24

        # Add the 2 least significant bits of the blue channel to the 8 bits of the red channel
        value = value | least_significant_bits

        return value

    def set_data_10b_rb(self, _value):
        value = self.convert_floatdata_to_10b_rb(_value)

        if value.ndim == 0:
            value = value.reshape((1, 1))
        elif value.ndim != 2:
            raise ValueError("Data should be a 2-D array or a scalar")
        if value.shape != self._data_shape:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                value.shape[1],
                value.shape[0],
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
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
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                value,
            )

    def get_data_enconding_10b_rb(self):
        with self.context:
            data = np.empty(self._data_shape, dtype="uint32")
            GL.glGetTextureImage(self.handle, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data.size * 4, data)

            # Extract the 10-bit values from the RGBA channels
            red_channel = (data >> 24) & 0xFF
            blue_channel = (data >> 8) & 0x3  # Extract the 2 least significant bits from the blue channel

            # Combine the channels to get the original 10-bit values
            combined_values = (red_channel << 2) | blue_channel

            # Convert back to float in the range [0, 1]
            float_data = combined_values.astype(np.float32) / 1023.0

            return float_data.reshape(self._data_shape)

    def get_data(self):
        with self.context:
            if self.encoding == "8b_r":
                data = np.empty(self._data_shape, dtype="float32")
                GL.glGetTextureImage(self.handle, 0, GL.GL_RED, GL.GL_FLOAT, data.size * 4, data)
                return data

            elif self.encoding == "10b_rb":
                return self.get_data_enconding_10b_rb()

            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")
