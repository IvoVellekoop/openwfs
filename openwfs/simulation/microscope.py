import numpy as np
import astropy.units as u
import scipy.ndimage
from astropy.units import Quantity
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve
from openwfs.simulation.mockdevices import MockImageSource, MockXYStage, MockCamera
from openwfs.slm import patterns


class Microscope:
    """A simulated microscope with pupil-conjugate SLM.

    The microscope simulates physical effects such as aberrations and noise, as well as devices typically found in a
    wavefront shaping microscope: a spatial light modulator, translation stages, and a camera.
    This simulation is designed to test algorithms for wavefront shaping and alignment.

    The simulation takes the field at the SLM, applies the phase aberrations, and masks the field with
    a pupil function corresponding to the numerical aperture of the microscope objective.
    This field is then Fourier transformed to obtain the intensity point spread function,
    with which the source image is convolved.

    Finally, the resulting image is mapped to the camera using a magnification factor, or affine transformation matrix.
    The camera object


    Here, the SLM image and the aberrations are

    TODO: It can be used with an actual OpenGL-based SLM object, so it also can be used to test the advanced functionality provided by that object.
    TODO: The configuration that is simulated where an SLM is conjugated to the back pupil of a microscope objective.
    All aberrations are considered to occur in the plane of that pupil.

    Attributes:
        source (ImageSource): Image source that produces 2-d images of the original 'specimen'
            as it is located in the focal plane.
            May be a MockImageSource (for a fixed image), or any other detector that produces 2-d data.
            The source must have `dimensions` specified in astropy distance units.
        magnification (float or matrix): magnification from object plane to camera.
            Can be a scalar, or a coordinate transformation matrix
            that maps points in the image plane to points in plane of the image sensor.
            See scipy.ndimage.affine_transform for the format of the matrix.
            numerical_aperture
        numerical_aperture (float): numerical aperture of the microscope objective.
            The field in the back pupil is cropped to this size
            (even if the slm and/or aberration map use a different NA).
        wavelength (astropy distance unit): wavelength of the light in micrometer.
            Used to compute the diffraction limit and the effect of aberrations.
        xy_stage (XYStage): Optional stage object that can be used to move the sample laterally.
            Defaults to a MockXYStage.
        z_stage (Stage): Optional stage object that moves the sample up and down to focus the microscope.
            Higher values are further away from the microscope objective.
            Defaults to a MockStage.
        slm (ImageSource): 2-D image containing the phase (in radians) as displayed on the SLM.
            The `dimensions` attribute must be set to twice the NA covered by the phase pattern.
            The pixels in the image are mapped to a square of -slm.NA(0) to +slm.NA(0) and -slm.NA(1) to slm.NA(1)
            at the back pupil pupil.
        aberrations (ImageSource): 2-D image containing the phase (in radians) of aberrations observed
            in the back pupil of the microscope objective.
            The `dimensions` attribute must be set to twice the NA covered by the aberration pattern.
        camera_resolution (tuple(float, float)): size of the image sensor in pixels.
            Defaults to 1024 pixels.
        camera_pixel_size (astropy distance unit): pixel pitch of the image sensor.
            Defaults to 10 μm.
        camera (ImageSource, output only): represents the camera in the magnified image plane of the microscope
            (i.e., where an actual camera would be located).
        truncation_factor (float): is used to simulate a gaussian beam illumination of the SLM/back pupil.
            Corresponds to w/r, with w the beam waist (1/e² intensity) and r the pupil radius.
            Defaults to None for a flat intensity distribution.

        Note:
            The aberration map and slm phase map are cropped/padded to the NA of the microscope objective, and
            scaled to have the same pixel resolution so that they can be added.

    """

    def __init__(self, source, magnification, numerical_aperture, wavelength: Quantity[u.nm],
                 xy_stage=None, z_stage=None, slm=None, aberrations=None, camera_resolution=(1024, 1024),
                 camera_pixel_size=10 * u.um, truncation_factor=0.0):
        self.source = source
        self.magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.wavelength = wavelength.to(u.nm)
        self.xy_stage = xy_stage or MockXYStage(0.1 * u.um, 0.1 * u.um)
        self.z_stage = z_stage  # or MockStage()
        self.slm = slm
        self.aberrations = aberrations
        self.truncation_factor = truncation_factor

        # create the mock camera object that will appear to capture the aberrated and translated image.
        # post-processors may be added to simulated physical effects like noise, bias, and saturation.
        self.camera = MockImageSource(data_shape=camera_resolution,
                                      pixel_size=camera_pixel_size,
                                      on_trigger=lambda: self._update())
        self.abbe_limit = 0.0
        self._pupil_resolution = 0.0
        self.psf = None

    def _update(self):
        """Updates the image on the camera sensor

        To compute the image:
        * First trigger the source, slm, and aberration sources
        * Then read the corresponding images.
        * Combines the slm and aberration images to compute the PSF
        * Crop the source image (not implemented yet) and upsample if needed (not implemented yet)
        * Convolve the source image with the PSF.
        * Compute the magnified and cropped image on the camera.
        """

        # Trigger source, slm and aberration sources
        self.source.trigger()
        if self.aberrations is not None:
            self.aberrations.trigger()
        if self.slm is not None:
            self.slm.trigger()
        s = self.source.read()  # read source image

        # Combine aberrations, slm pattern and pupil to compute the PSF
        #
        # First, calculate the magnification and field of view.
        m = self.magnification if np.isscalar(self.magnification) else np.sqrt(np.linalg.det(self.magnification))
        fov = self.camera.pixel_size * np.max(self.camera.data_shape) / m

        # Then, calculate the required resolution for the pupil map.
        # The coordinates in the pupil plane correspond to spatial frequencies in the image plane.
        # For a normalized pupil coordinate ξ, we have a spatial frequency of k_x = ξ k_0.
        # A maximum spatial frequency of NA·k_0 corresponds to a period of λ/NA, which gives
        # a Nyquist limit of λ/(2.0·NA), which is the Abbe diffraction limit.
        # The total field of view holds 2.0 Δ NA / λ diffraction limited points,
        # which means that we need exactly that many points in the NA.
        self.abbe_limit = 0.5 * self.wavelength / self.numerical_aperture
        self._pupil_resolution = int(np.ceil(float(fov / self.abbe_limit)))
        print(self._pupil_resolution)
        pupil_field = patterns.disk(self._pupil_resolution) if self.truncation_factor is None else \
            patterns.gaussian(self._pupil_resolution, 1.0 / self.truncation_factor)
        if self.aberrations is not None and self.slm is not None:
            pupil_field *= np.exp(1.0j * (self._read_crop(self.aberrations) + self._read_crop(self.slm)))
        elif self.slm is not None:
            pupil_field *= np.exp(1.0j * self._read_crop(self.slm))
        elif self.aberrations is not None:
            pupil_field *= np.exp(1.0j * self._read_crop(self.aberrations))

        # finally, pad the pupil field so that the diameter of the pupil field
        # corresponds to a focus with the size of a single pixel in the source image
        # - first compute the ratio of Abbe limit (i.e. the resolution corresponding to the current pupil size)
        #   to the pixel size of the source image (i.e. the resolution corresponding to the padded pupil size)
        # - then pad the pupil field so that the size gets multiplied by that ratio
        pupil_resolution = round(float(self.abbe_limit / self.source.pixel_size) * self._pupil_resolution)
        if pupil_resolution < self._pupil_resolution:
            raise Exception("Pixel size in the source image is larger than the Abbe diffraction limit")

        # compute the PSF
        psf = np.abs(np.fft.fft2(np.fft.ifftshift(pupil_field), (pupil_resolution, pupil_resolution))) ** 2
        psf = np.fft.ifftshift(psf) / np.sum(psf)
        self.psf = psf
        s = fftconvolve(s, psf, 'same')

        # transform image size and orientation to camera
        m = self.magnification * (self.source.pixel_size / self.camera.pixel_size).to_value(u.dimensionless_unscaled)
        if np.isscalar(m):
            m = np.eye(3) * m
            m[2, 2] = 1
        offset = np.eye(3)
        offset[0, 2] += self.xy_stage.x / self.source.pixel_size
        offset[1, 2] += self.xy_stage.y / self.source.pixel_size
        m = m @ offset  # apply offset first, then magnification

        return affine_transform(s, np.linalg.inv(m), order=1, output_shape=self.camera.data_shape)

    def _read_crop(self, source):
        """crop/pad an image to the NA of the microscope objective and scale to the internal resolution"""

        img = source.read()
        pixel_size = img.pixel_size.value()  # size in normalized NA coordinates

        # scale the image
        scale = pixel_size / (self.numerical_aperture / self._pupil_resolution)
        return affine_transform(img, scale, output_shape=(self._pupil_resolution, self._pupil_resolution), order=0)
