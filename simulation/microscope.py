import numpy as np
import pint
from slm import patterns
from scipy.ndimage import affine_transform

ureg = pint.UnitRegistry()


class MockImageSource:
    """'Camera' that dynamically generates images using an 'on_trigger' callback.
    Note that this object does not implement the full camera interface. Does not support resizing yet"""

    @staticmethod
    @ureg.check(None, ureg.um)
    def from_image(image, pixel_size):
        """Create a MockImageSource that displays the given numpy array as image.
        :param image : 2-d numpy array to return on 'read'
        :param pixel_size : size of a single pixel, must have a pint unit of type length
        """
        return MockImageSource(lambda i: image, image.shape, pixel_size)

    @ureg.check(None, None, None, ureg.um)
    def __init__(self, on_trigger, data_shape, pixel_size):
        self._on_trigger = on_trigger
        self._image = np.empty(data_shape, dtype="float32")
        self._measurement_time = 0  # property may be set to something else to mimic acquisition time
        self._triggered = 0  # currently only support a buffer with a single frame
        self.pixel_size = pixel_size
        self.data_shape = data_shape
        self.measurement_time = 0.0

    def trigger(self):
        if self._triggered != 0:
            raise RuntimeError('Buffer overflow: this camera can only store 1 frame, and the previous frame was not '
                               'read yet')
        self._image = self._on_trigger(self._image)
        self._triggered += 1

    def read(self):
        if self._triggered == 0:
            raise RuntimeError('Buffer underflow: trying to read an image without triggering the camera')

        return self._image


class Microscope:
    """A simulated microscope.
    The microscope simulates physical effects such as aberrations and noise, as well as devices typically found in a
    wavefront shaping microscope: a spatial light modulator, translation stages, and a camera.

    This simulation is designed to test algorithms for wavefront shaping, alignment calibration,
    and calibrating the lookup table of the SLM. It can be used with an actual OpenGL-based SLM object,
    so it also can be used to test the advanced functionality provided by that object.

    The configuration that is simulated where an SLM is conjugated to the back pupil of a microscope objective.
    All aberrations are considered to occur in the plane of that pupil.
    """

    @ureg.check(None, None, None, None, ureg.nm, ureg.um, None, None, None)
    def __init__(self, source, m, na, wavelength, pixel_size, stage=None, slm=None, aberrations=None):
        """
        :param source: detector object producing 2-dimensional images of the object to be imaged.

        :param m: magnification from object plane to camera. Can be a scalar, or a coordinate transformation matrix
        that maps points in the image plane to points in plane of the image sensor. See
        scipy.ndimage.affine_transform for the format of the matrix.

        :param na: numerical aperture of the microscope objective

        :param wavelength: wavelength of the light in micrometer

        :param pixel_size: size of the pixels on the camera object that represents the microscope image.

        :param stage: mock stage.

        :param slm: mock slm, or actual OpenGL-based SLM object. May include an amplitude map of the illumination
        profile at the SLM surface. The SLM is expected to be calibrated to normalized pupil coordinates, where
        (0,0) is located on the optical axis, and the distance to the center corresponds to n sin(θ). This means that,
        in the image plane, pupil coordinates (px,py) map to propagation vectors: k_x = x k_0, k_y = y k_0.

        :param aberrations: wavefront distortion in the image plane (in radians). The aberration map is applied to
        the at current active phase patch of the SLM, which typically is a square ranging from -NA to +NA in
        normalized pupil coordinates, i.e. the image spans the full back pupil.
        """
        self.source = source
        self.M = m
        self.NA = na
        self.wavelength = wavelength
        self.slm = slm
        self.stage = stage
        self.aberrations = aberrations

        # construct pupil mask
        if aberrations is None:
            self.pupil = patterns.disk(500, na)
        else:
            if aberrations.shape(0) != aberrations.shape(1):
                ValueError("aberration map should be square")
            self.pupil = patterns.disk(aberrations.shape(0), na) * np.exp(1j * aberrations)

        # create mock camera object that will appear to capture the aberrated and translated image.
        # post-processors may be added to simulated physical effects like noise, bias, and saturation.
        self.camera = MockImageSource(data_shape=self.source.data_shape,
                                      pixel_size=pixel_size,
                                      on_trigger=lambda i: self._update(i))

    def _update(self, image):
        # compute point spread function
        # transform image size and orientation to camera
        # convolve with point spread function
        self.source.trigger()
        s = self.source.read()
        m = self.M * (self.source.pixel_size / self.camera.pixel_size).to_base_units().magnitude
        if np.isscalar(m):
            m = np.ones(2) * m
        affine_transform(s, m, output=image, order=1)
        return image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.maximum(np.random.randint(-10000, 100, (500, 500), dtype=np.int16), 0)
    src = MockImageSource.from_image(img, 100 * ureg.nm)
    mic = Microscope(src, m=10, na=0.85, wavelength=532.8 * ureg.nm, pixel_size=6.45 * ureg.um)

    c = mic.camera
    c.trigger()
    cim = c.read()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(cim)
    plt.show()
