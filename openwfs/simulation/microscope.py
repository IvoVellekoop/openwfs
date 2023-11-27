import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.signal import fftconvolve
from typing import Union
from ..simulation.mockdevices import MockXYStage, MockCamera
from ..slm import patterns
from ..core import Processor, get_pixel_size, set_pixel_size
from ..utilities import project, place, imshow


class Microscope(Processor):
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
                 camera_pixel_size=10 * u.um, truncation_factor=None, analog_max=0.0):
        """

        Args:
            source (ImageSource): Image source that produces 2-d images of the original 'specimen'
                as it is located in the focal plane.
                Could be a MockSource (for a fixed image), or any other detector that produces 2-d data.
                The source must have `dimensions` specified in astropy distance units.
            magnification:
            numerical_aperture:
            wavelength:
            xy_stage:
            z_stage:
            slm (ImageSource): 2-D image containing the phase (in radians) as displayed on the SLM.
                The `dimensions` attribute must be set to twice the NA covered by the phase pattern.
                The pixels in the image are mapped to a square of -slm.NA(0) to +slm.NA(0) and -slm.NA(1) to slm.NA(1)
                at the back pupil.
            aberrations (ImageSource): 2-D image containing the phase (in radians) of aberrations observed
                in the back pupil of the microscope objective.
                The `dimensions` attribute must be set to twice the NA covered by the aberration pattern.
            camera_resolution (tuple(float, float)): size of the image sensor in pixels.
                Defaults to 1024 pixels.
            camera_pixel_size (astropy distance unit): pixel pitch of the image sensor.
                Defaults to 10 μm.
            truncation_factor:
        """
        super().__init__(source, aberrations, slm, data_shape=camera_resolution, pixel_size=camera_pixel_size)
        self._magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.aberration_transform = None  # coordinate transform from the `aberration` object to the pupil plane
        self.slm_transform = None  # coordinate transform from the `slm` object to the pupil plane
        self.wavelength = wavelength.to(u.nm)
        self.oversampling_factor = 2.0
        self.xy_stage = xy_stage or MockXYStage(0.1 * u.um, 0.1 * u.um)
        self.z_stage = z_stage  # or MockStage()
        self.truncation_factor = truncation_factor
        self.camera = MockCamera(self, analog_max=analog_max)
        self.psf = None
        self.slm = slm
        assert source is not None

    def _fetch(self, out: Union[np.ndarray, None], source: np.ndarray, aberrations: np.ndarray,  # noqa
               slm: np.ndarray) -> np.ndarray:
        """Updates the image on the camera sensor

        To compute the image:
        * First trigger the source, slm, and aberration sources
        * Then read the corresponding images.
        * Combines the slm and aberration images to compute the PSF
        * Crop the source image (not implemented yet) and upsample if needed (not implemented yet)
        * Convolve the source image with the PSF.
        * Compute the magnified and cropped image on the camera.
        """

        # First crop and downscale the source image to have the same size as the output
        # todo: add some padding
        # todo: add option for oversampling
        source_pixel_size = get_pixel_size(source)
        target_pixel_size = self.pixel_size / self.magnification
        if np.any(source_pixel_size >= target_pixel_size):
            raise Exception("The resolution of the specimen image is worse than that of the output.")

        # construct matrix for translation of the specimen
        source = place(self.data_shape, target_pixel_size, source, Quantity((self.xy_stage.y, self.xy_stage.x)))

        # Calculate the field in the pupil plane.
        #
        # First, set up pupil coordinates such that:
        # 1. the Fourier transform of the pupil has a resolution
        #    that exactly matches the resolution of the specimen image.
        # 2. the resolution in the pupil plane is high enough such that
        #    the Fourier transform of the pupil field has a size that is at least equal to the fov of the microscope.
        #    This means that then number of pixels should be at least as high as the number of points in the fov
        #    (at the resolution of the specimen image).
        #
        # The NA of the pupil corresponds to a disk that is contained in this pupil plane.
        # We compute the aberrations over the full pupil plane, and clip to the NA by using a disk function.
        #

        # condition 1. Extent of pupil in pupil coordinates: Abbe limit should give pixel_size resolution
        pupil_extent = self.wavelength / target_pixel_size

        # condition 2. Minimum number of pixels in x and y
        pupil_shape = self.data_shape
        pupil_pixel_size = pupil_extent / pupil_shape

        # Compute the field in the pupil plane
        # Aberrations and the SLM phase are mapped to the pupil plane coordinates
        if self.truncation_factor is None:
            pupil_field = patterns.disk(pupil_shape, radius=self.numerical_aperture, extent=pupil_extent)
        else:
            pupil_field = patterns.gaussian(pupil_shape, waist=self.truncation_factor * self.numerical_aperture,
                                            truncation_radius=self.numerical_aperture, extent=pupil_extent)

        if aberrations is not None and slm is not None:
            pupil_field *= np.exp(1.0j *
                                  project(pupil_shape, pupil_pixel_size, aberrations, self.aberration_transform) +
                                  project(pupil_shape, pupil_pixel_size, slm, self.slm_transform))
        elif slm is not None:
            pupil_field *= np.exp(1.0j * project(pupil_shape, pupil_pixel_size, slm, self.slm_transform))
        elif aberrations is not None:
            pupil_field *= np.exp(1.0j * project(pupil_shape, pupil_pixel_size, aberrations, self.aberration_transform))

        # Compute the point spread function
        # This is done by Fourier transforming the pupil field and taking the absolute value squared
        # Due to condition 1, after the Fourier transform,
        # the pixel size matches that of the source (the specimen image).
        # Note: there is no need to `ifftshift` the pupil field, since we are taking the absolute value anyway
        psf = np.abs(np.fft.fft2(pupil_field)) ** 2
        psf = np.fft.fftshift(psf) / np.sum(psf)
        self.psf = psf  # store psf for later inspection

        # todo: test if the convolution does not introduce an offset
        source = fftconvolve(source, psf, 'same')
        # apply magnification by just adjusting the pixel size
        # note, this is not needed as it happens automatically in the _do_fetch function
        source = set_pixel_size(source, self.pixel_size)

        if out is None:
            out = source
        else:
            out[...] = source
        return out

    @property
    def magnification(self) -> float:
        """Magnification from object plane to image plane.

        Note that, as in a real microscope, the magnification does not affect the effective resolution of the image.
        The resolution is determined by the Abbe diffraction limit λ/2NA.
        """
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        self._magnification = value

    @property
    def abbe_limit(self) -> Quantity:
        """Returns the Abbe diffraction limit: λ/(2 NA)"""
        return 0.5 * self.wavelength / self.numerical_aperture
