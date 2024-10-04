import warnings
from typing import Optional, Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve

from ..core import Processor, Detector
from ..plot_utilities import imshow  # noqa - for debugging
from ..simulation.mockdevices import XYStage, StaticSource
from ..utilities import project, place, Transform, get_pixel_size, patterns


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
    The propagation is normalized such that a pupil fully filled with a field strength of 1.0 will produce an image
    that has the same total intensity as the source image.
    """

    def __init__(
        self,
        source: Union[Detector, np.ndarray],
        *,
        data_shape=None,
        numerical_aperture: float = 1.0,
        wavelength: Quantity[u.nm],
        magnification: float = 1.0,
        xy_stage=None,
        z_stage=None,
        incident_field: Union[Detector, ArrayLike, None] = None,
        incident_transform: Optional[Transform] = None,
        aberrations: Union[Detector, np.ndarray, None] = None,
        aberration_transform: Optional[Transform] = None,
        multi_threaded: bool = True
    ):
        """
        Args:
            source: 2-D image (must have `pixel_size` metadata), or
                a detector that produces 2-D images of the original 'specimen'
            data_shape: shape (size in pixels) of the output.
                Default value: source.data_shape
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
            incident_field: Produces 2-D complex images containing the field output of the SLM.
                If no `slm_transform` is specified, the `pixel_size` attribute should
                 correspond to normalized pupil coordinates
                (e.g. with a disk of radius 1.0, i.e. an extent of 2.0, corresponding to an NA of 1.0)
            incident_transform (Optional[Transform]):
                Optional Transform that transforms the phase pattern from the slm object
                (in slm.pixel_size units) to normalized pupil coordinates.
                Typically, the slm image is already in normalized pupil coordinates,
                but this transform can be used to mimic SLM misalignment.
            aberrations: 2-D image containing the phase (in radians) of aberrations observed
                in the back pupil of the microscope objective, or a Detector object that automatically produces such
                images. The `extent` attribute corresponds to normalized pupil coordinates. For example, with a
                numerical aperture of 0.6, the extent of the image should be 1.2. If a 2-D image without pixel_size
                metadata is provided, the extent is automatically set to 2.0 * numerical_aperture.
            aberration_transform (Optional[Transform]):
                Optional Transform that transforms the phase pattern from the aberration object
                (in slm.pixel_size units) to normalized pupil coordinates.
                Typically, the slm image is already in normalized pupil coordinates,
                but this transform may e.g., be used to scale an aberration pattern
                from extent 2.0 to 2.0 * NA.

        Note:
            The aberration map and slm phase map are cropped/padded to the NA of the microscope objective, and
            scaled to have the same pixel resolution so that they can be added.
        """
        if not isinstance(source, Detector):
            if get_pixel_size(source) is None:
                raise ValueError("The source must have a pixel_size attribute.")
            source = StaticSource(source)

        if aberrations is not None and not isinstance(aberrations, Detector):
            if get_pixel_size(aberrations) is None:
                aberrations = StaticSource(aberrations)

        super().__init__(source, aberrations, incident_field, multi_threaded=multi_threaded)
        self._magnification = magnification
        self._data_shape = data_shape if data_shape is not None else source.data_shape
        self.numerical_aperture = numerical_aperture
        self.aberration_transform = aberration_transform
        self.slm_transform = incident_transform
        self.wavelength = wavelength.to(u.nm)
        self.oversampling_factor = 2.0
        self.xy_stage = xy_stage or XYStage(0.1 * u.um, 0.1 * u.um)
        self.z_stage = z_stage  # or MockStage()
        self._psf = None

    def _fetch(
        self,
        source: np.ndarray,
        aberrations: np.ndarray,  # noqa
        incident_field: np.ndarray,
    ) -> np.ndarray:
        """
        Updates the image on the camera sensor

        To compute the image:
        * First trigger the source, slm, and aberration sources
        * Then read the corresponding images.
        * Combines the slm and aberration images to compute the PSF
        * Crop the source image (not implemented yet) and upsample if needed (not implemented yet)
        * Convolve the source image with the PSF.
        * Compute the magnified and cropped image on the camera.

        Args:
            source:
            aberrations:
            incident_field:

        Returns:

        """

        # First crop and downscale the source image to have the same size as the output
        # todo: add some padding
        # todo: add option for oversampling
        source_pixel_size = get_pixel_size(source)
        target_pixel_size = self.pixel_size / self.magnification
        if np.any(source_pixel_size > target_pixel_size):
            warnings.warn("The resolution of the specimen image is worse than that of the output.")

        # Note: there seems to be a bug (feature?) in `fftconvolve` that shifts the image by one pixel
        # when the 'same' option is used. To compensate for this feature,
        # the image is shifted by `-source_pixel_size` here.
        # TODO: this seems to add an emtpy row and column to the image, which is not what we want.
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

        # Compute the field in the pupil plane
        # The aberrations and the SLM phase pattern are both mapped to the pupil plane coordinates
        pupil_field = patterns.disk(pupil_shape, radius=self.numerical_aperture, extent=pupil_extent)
        pupil_area = np.sum(pupil_field)  # TODO (efficiency): compute area directly from radius

        # Project aberrations
        if aberrations is not None:
            # use default of 2.0 * NA for the extent of the aberration map if no pixel size is provided
            aberration_extent = (2.0 * self.numerical_aperture,) * 2 if get_pixel_size(aberrations) is None else None
            pupil_field = pupil_field * np.exp(
                1.0j
                * project(
                    aberrations,
                    source_extent=aberration_extent,
                    out_extent=pupil_extent,
                    out_shape=pupil_shape,
                    transform=self.aberration_transform,
                )
            )

        # Project SLM fields
        if incident_field is not None:
            pupil_field = pupil_field * project(
                incident_field,
                out_extent=pupil_extent,
                out_shape=pupil_shape,
                transform=self.slm_transform,
            )

        # Compute the point spread function
        # This is done by Fourier transforming the pupil field and taking the absolute value squared
        # Due to condition 1, after the Fourier transform,
        # the pixel size matches that of the source (the specimen image).
        # Note: there is no need to `ifftshift` the pupil field, since we are taking the absolute value anyway
        psf = np.abs(np.fft.ifft2(pupil_field)) ** 2
        psf = np.fft.ifftshift(psf) * (psf.size / pupil_area)
        self._psf = psf  # store psf for later inspection

        return fftconvolve(source, psf, "same")

    @property
    def magnification(self) -> float:
        """
        Magnification from object plane to image plane.

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

    @property
    def pixel_size(self) -> Quantity:
        """Returns the pixel size in the image plane
        This value is always equal to `source.pixel_size * magnification`"""
        return self._sources[0].pixel_size * self.magnification

    @property
    def data_shape(self):
        """Returns the shape of the image in the image plane"""
        return self._data_shape
