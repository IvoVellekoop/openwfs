import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy.signal import fftconvolve
from typing import Union, Optional
from ..simulation.mockdevices import MockXYStage, MockCamera
from ..slm import patterns
from ..core import Processor, Detector, get_pixel_size, set_pixel_size
from ..utilities import project, place, Transform, imshow
from ..processors import TransformProcessor


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
        camera (ImageSource, output only): represents the camera in the magnified image plane of the microscope
            (i.e., where an actual camera would be located).
        truncation_factor (float): is used to simulate a gaussian beam illumination of the SLM/back pupil.
            Corresponds to w/r, with w the beam waist (1/e² intensity) and r the pupil radius.
            Defaults to None for a flat intensity distribution.

     numerical_aperture (float):
                numerical
            magnification:
            wavelength:
            xy_stage:
            z_stage:
            camera_resolution (tuple(float, float)): size of the image sensor in pixels.
                Defaults to 1024 pixels.
            camera_pixel_size (astropy distance unit): pixel pitch of the image sensor.
                Defaults to 10 μm.
            truncation_factor:
        Note:
            The aberration map and slm phase map are cropped/padded to the NA of the microscope objective, and
            scaled to have the same pixel resolution so that they can be added.

    """

    def __init__(self, source: Detector, *, data_shape=None, numerical_aperture: float, wavelength: Quantity[u.nm],
                 magnification: float = 1.0, xy_stage=None, z_stage=None,
                 slm: Optional[Detector] = None, slm_transform: Optional[Transform] = None,
                 aberrations: Optional[Detector] = None, aberration_transform: Optional[Transform] = None,
                 truncation_factor: Optional[float] = None, pixel_size: Quantity = None):
        """
        Args:
            source (Detector): Detector that produces 2-d images of the original 'specimen'
            data_shape: shape (size in pixels) of the output.
                Default value: source.data_shape
            pixel_size (Quantity): 1 or 2-element pixel size (in astropy units).
                Default value: source.pixel_size * magnification
            numerical_aperture: Numerical aperture of the microscope objective
            wavelength: Wavelength of the light used for imaging,
                the wavelength and numerical_aperture together determine the resolution of the microscope.
            magnification: Scalar magnification factor between input and output image.
                Note that this factor does not affect the effective image resolution.
                Increasing the magnification will just produce a zoomed-in blurred image.
                The default settings for data_shape, pixel_size and magnification cause the simulated microscope
                to only convolve the source image with the point-spread function (PSF) without applying any scaling.
            xy_stage (XYStage): Optional stage object that can be used to move the sample laterally.
                Defaults to a MockXYStage.
            z_stage (Stage): Optional stage object that moves the sample up and down to focus the microscope.
                Higher values are further away from the microscope objective.
                Defaults to a MockStage.
            slm (Detector): Detector that produces 2-d images containing the phase (in radians) as displayed on the SLM.
                If no `slm_transform` is specified, the `pixel_size` attribute should correspond to normalized pupil coordinates
                (e.g. with a disk of radius 1.0, i.e. an extent of 2.0, corresponding to an NA of 1.0)
            slm_transform (Transform|None):
                Optional Transform that transforms the phase pattern from the slm object (in slm.pixel_size units) to normalized pupil
                coordinates.
                Typically, the slm image is already in normalized pupil coordinates,
                but this transform can be used to mimic SLM misalignment.
            aberrations (ImageSource): 2-D image containing the phase (in radians) of aberrations observed
                in the back pupil of the microscope objective.
                The `pixel_size` attribute corresponds to normalized pupil coordinates.
            aberration_transform (Transform|None):
                Optional Transform that transforms the phase pattern from the aberration object (in slm.pixel_size units) to normalized pupil
                coordinates.
                Typically, the slm image is already in normalized pupil coordinates,
                but this transform may e.g. be used to scale an aberration pattern
                from extent 2.0 to extent 2.0 * NA.
            truncation_factor (float | None):
                When set to a value > 0.0,
                the microscope pupil is illuminated by a Gaussian beam (instead of a flat intensity).
                The value corresponds to the beam waist relative to the full pupil (the full NA).
        """
        if pixel_size is None:
            pixel_size = source.pixel_size * magnification
        if not isinstance(source, Detector):
            raise ValueError("`source` should be a Detector object.")

        super().__init__(source, aberrations, slm, pixel_size=pixel_size, data_shape=data_shape)
        self._magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.aberration_transform = aberration_transform
        self.slm_transform = slm_transform
        self.wavelength = wavelength.to(u.nm)
        self.oversampling_factor = 2.0
        self.xy_stage = xy_stage or MockXYStage(0.1 * u.um, 0.1 * u.um)
        self.z_stage = z_stage  # or MockStage()
        self.truncation_factor = truncation_factor
        self.slm = slm
        self._psf = None

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
        if np.any(source_pixel_size > target_pixel_size):
            raise Exception("The resolution of the specimen image is worse than that of the output.")

        # construct matrix for translation of the specimen
        # Note: there seems to be a bug (feature?) in fftconvolve that shifts the image by one pixel
        # when the 'same' option is used.
        # To compensate for this feature, the image is shifted by - source_pixel_size here.
        # this will cause an empty line at the side of the image!
        # todo: implement our own version of fftconvolve, or crop manually
        shift = Quantity((self.xy_stage.y, self.xy_stage.x)) - source_pixel_size
        source = place(self.data_shape, target_pixel_size, source, shift)

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
        # TODO: think about what happens when the requested output resolution is lower than the diffraction limit
        #       at the moment, not the full pupil is used.
        # TODO: think about what happens when the slm is smaller than the pupil

        # condition 1. Extent of pupil in pupil coordinates: Abbe limit should give pixel_size resolution
        pupil_extent = self.wavelength / target_pixel_size

        # condition 2. Minimum number of pixels in x and y should be data_shape
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
            pupil_field = pupil_field * np.exp(1.0j *
                                               project(pupil_shape, pupil_pixel_size, aberrations,
                                                       self.aberration_transform) +
                                               project(pupil_shape, pupil_pixel_size, slm, self.slm_transform))
        elif slm is not None:
            pupil_field = pupil_field * np.exp(1.0j * project(pupil_shape, pupil_pixel_size, slm, self.slm_transform))
        elif aberrations is not None:
            pupil_field = pupil_field * np.exp(
                1.0j * project(pupil_shape, pupil_pixel_size, aberrations, self.aberration_transform))

        # Compute the point spread function
        # This is done by Fourier transforming the pupil field and taking the absolute value squared
        # Due to condition 1, after the Fourier transform,
        # the pixel size matches that of the source (the specimen image).
        # Note: there is no need to `ifftshift` the pupil field, since we are taking the absolute value anyway
        psf = np.abs(np.fft.ifft2(pupil_field)) ** 2
        psf = np.fft.ifftshift(psf) / np.sum(psf)
        self._psf = psf  # store psf for later inspection

        # todo: test if the convolution does not introduce an offset
        source = fftconvolve(source, psf, 'same')
        # apply magnification by just adjusting the pixel size
        # note, this is not needed as it happens automatically in the _do_fetch function
        source = set_pixel_size(source, self.pixel_size)
        print("Microscope image fetched\n")
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

    def get_camera(self, *, transform: Union[Transform, None] = None, **kwargs) -> Detector:
        """Returns a simulated camera that observes the microscope image.

        The camera is a MockCamera object that simulates an AD-converter with optional noise.
        shot noise and readout noise (see MockCamera for options).
        In addition to the inputs accepted by the MockCamera constructor (data_shape, analog_max, shot_noise, etc.),
        it is also possible to specify a transform, to mimic the (mis)alignment of the camera.
        """
        if transform is None:
            src = self
        else:
            src = TransformProcessor(self, transform)

        return MockCamera(src, **kwargs)
