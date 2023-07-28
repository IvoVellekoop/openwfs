import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve
from .mockdevices import MockImageSource, MockXYStage, MockCamera
from openwfs.slm import patterns


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

    def __init__(self, source, m, na, wavelength: Quantity[u.nm], pixel_size: Quantity[u.um], stage=None,
                 slm=None, aberrations=None):
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
        self.wavelength = wavelength.to(u.nm)
        self.slm = slm
        self.stage = stage if stage is not None else MockXYStage(0.1 * u.um, 0.1 * u.um)
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
        self.camera = MockCamera(MockImageSource(data_shape=self.source.data_shape,
                                                 pixel_size=pixel_size,
                                                 on_trigger=lambda i: self._update(i)))

    def _update(self, image):
        # transform image size and orientation to camera
        # compute point spread function
        # convolve with point spread function
        self.source.trigger()
        s = self.source.read()
        m = self.M * (self.source.pixel_size / self.camera.pixel_size).to_value(u.dimensionless_unscaled)
        if np.isscalar(m):
            m = np.eye(3) * m
            m[2, 2] = 1

        offset = np.eye(3)
        offset[0, 2] += self.stage.x / self.source.pixel_size
        offset[1, 2] += self.stage.y / self.source.pixel_size
        m = m @ offset  # apply offset first, then magnification

        affine_transform(s, m, output=image, order=1)

        #compute the PSF (todo)
        psf = np.zeros(image.shape)
        psf[0, 0] = 1
        psf = np.fft.ifftshift(psf)

        return fftconvolve(image, psf, 'same')
