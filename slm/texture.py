import numpy as np
import weakref
from OpenGL.GL import *


class Texture:
    def __init__(self, context, texture_type=GL_TEXTURE_2D):
        self.context = weakref.ref(context)
        self.handle = glGenTextures(1)
        self.type = texture_type
        self.synchronized = False  # self.data is not yet synchronized with texture in GPU memory
        self._data = None
        self._data_size_changed = True  # when True, synchronize() creates a new texture instead of overwriting the old
        self.data = 0  # create a single pixel texture as default
        glBindTexture(self.type, self.handle)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_WRAP_R, GL_REPEAT)
        glTexParameteri(self.type, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(self.type, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    def __del__(self):
        if self.context() is not None and hasattr(self, 'handle'):
            self.context().activate()
            glDeleteTextures(1, [self.handle])

    @property
    def data(self):
        """ Returns the data array currently associated with the texture. If synchronized _was_ True, this is also the
            data that is currently on the GPU. Note: calling this function resets the 'synchronized' flag and causes the
            texture data to be uploaded to the GPU during the next slm.update. This is to enable a common use case where
            the array is modified in place, e.g. 'texture.data[1,1]=0'."""
        self.synchronized = False
        return self._data

    @data.setter
    def data(self, value):
        """ Set texture data. The dimensionality of the data should match that of the texture.
            As an exception, all texture types can be set to a scalar value.
            Note that textures holds a reference to the array that was last stored in it. A copy is only
            made when the array is not a numpy float32 array yet. If the data in the referenced array is modified,
            the data on the GPU and the data in 'phases' are out of sync. They will be synchronized automatically when
            the slm is updated (drawn)."""
        value = np.array(value, dtype=np.float32, copy=False)

        # check if data has correct dimension, convert scalars to arrays of correct dimension
        if self.type == GL_TEXTURE_1D:
            if value.ndim == 0:
                value.shape = 1
            elif value.ndim != 1:
                raise ValueError("Data should be 1-d array")

        elif self.type == GL_TEXTURE_2D:
            if value.ndim == 0:
                value.shape = (1, 1)
            elif value.ndim != 2:
                raise ValueError("Data should be 2-d array")

        else:
            raise ValueError("Texture type not supported")

        self._data_size_changed |= (self._data is None or self._data.shape != value.shape)
        self._data = value  # store data
        self.synchronized = False

    def synchronize(self):
        """Copies the data from main memory to the GPU. This function is called automatically during slm.update(),
        so there is no need to call it directly."""
        if self.synchronized:
            return

        self.context().activate()
        (internal_format, data_format, data_type) = (GL_R32F, GL_RED, GL_FLOAT)
        glBindTexture(self.type, self.handle)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # alignment is at least 4 bytes since we are using float32 for everything

        if self.type == GL_TEXTURE_1D:
            if self._data_size_changed:
                glTexImage1D(GL_TEXTURE_1D, 0, internal_format, self.data.shape[0], 0, data_format, data_type,
                             self.data)
            else:
                glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 0, self.data.shape[0], data_format, data_type, self.data)

        elif self.type == GL_TEXTURE_2D:
            if self._data_size_changed:
                glTexImage2D(GL_TEXTURE_2D, 0, internal_format, self.data.shape[0], self.data.shape[1], 0, data_format,
                             data_type, self.data)
            else:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.data.shape[0], self.data.shape[1], data_format, data_type,
                                self.data)

        self.synchronized = True
