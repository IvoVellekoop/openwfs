import warnings
from typing import Optional, Union

import astropy.units as u
import cv2
import numpy as np
from astropy.units import Quantity
from scipy.signal import fftconvolve

from openwfs.utilities.utilities import set_extent, get_extent

from ..core import Processor, Detector
from ..plot_utilities import imshow  # noqa - for debugging
from ..simulation.mockdevices import XYStage, LinearStage, StaticSource
from ..utilities import project, place, Transform, get_pixel_size, patterns, get_extent
from ..utilities.patterns import propagation


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
        source: Detector,
        *,
        data_shape=None,
        numerical_aperture: float = 1.0,
        wavelength: Quantity[u.nm],
        nonlinearity: int = 1,
        xy_stage=None,
        z_stage=None,
        immersion_refractive_index: Optional[float] = 1.0,
        incident_field: Detector | None = None,
        incident_transform: Optional[Transform] = None,
        aberrations: Detector | None = None,
        aberration_transform: Optional[Transform] = None,
        multi_threaded: bool = True,
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
            nonlinearity: Exponent to which the PSF is raised. This can be used to simulate two-photon microscopy (nonlinearity=2),
                or multiphoton microscopy in general (nonlinearity > 2).
            xy_stage (XYStage): Optional stage object that can be used to move the sample laterally.
                Defaults to a MockXYStage.
            z_stage (Stage): Optional stage object that moves the sample up and down to focus the microscope.
                Higher values are further away from the microscope objective.
                Defaults to a MockStage.
            immersion_refractive_index: The refractive index of the immersion medium.
            incident_field: Produces 2-D complex images containing the field output of the SLM.
                If no `slm_transform` is specified, the `pixel_size` attribute should
                 correspond to normalized pupil coordinates
                (e.g. with a disk of radius 1.0, i.e. an extent of 2.0, corresponding to an NA of 1.0)
            incident_transform (Optional[Transform]):
                Optional Transform that transforms the phase pattern from the slm object
                (in slm.pixel_size units) to normalized pupil coordinates.
                Typically, the slm image is already in normalized pupil coordinates,
                but this transform can be used to mimic SLM misalignment.
                Default if no transform is provided: Transform(np.diag(2 / (incident_field.pixel_size * incident_field.data_shape)))
                such that the incident is assumed to have an extent of 2.0 in normalized pupil coordinates.
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
            raise ValueError("The source must be a Detector object.")

        if aberrations is not None and not isinstance(aberrations, Detector):
            raise ValueError("The aberrations must be a Detector object or None.")

        # First crop and downscale the source image to have the same size as the output
        # todo: add some padding
        # todo: add option for oversampling
        source_pixel_size = source.pixel_size

        self.aberration_transform = aberration_transform
        # if no transform is provided, assume that the incident field is already in normalized pupil coordinates
        self._incident_transform = (
            incident_transform
            if incident_transform is not None or incident_field is None
            else Transform(np.diag(2 / (incident_field.pixel_size * incident_field.data_shape)))
        )

        self.xy_stage = xy_stage or XYStage(0.1 * u.um, 0.1 * u.um)
        self.z_stage = z_stage or LinearStage(0.1 * u.um)
        output_shape = data_shape if data_shape is not None else source.data_shape

        domain_extent = wavelength / self.pixel_size / numerical_aperture

        self._Pupil_Field = _Pupil_Field(
            pupil_shape=output_shape,
            pupil_extent=domain_extent,
            incident_field=incident_field,
            incident_transform=incident_transform,
            aberrations=aberrations,
            aberration_transform=aberration_transform,
            multi_threaded=multi_threaded,
        )
        self.pupil_field = _Propagator(
            pupil_field=self._Pupil_Field,
            pupil_shape=output_shape,
            pupil_extent=domain_extent,
            aberrations=aberrations,
            incident_field=incident_field,
            wavelength=wavelength,
            numerical_aperture=numerical_aperture,
            immersion_refractive_index=immersion_refractive_index,
            incident_transform=incident_transform,
            aberration_transform=aberration_transform,
            xy_stage=xy_stage,
            z_stage=z_stage,
        )
        # PSF of the microscope, which is used to convolve the source image
        self.psf = _PSF(
            pupil_field=self.pupil_field,
            data_shape=output_shape,
            pupil_extent=domain_extent,
            numerical_aperture=numerical_aperture,
            wavelength=wavelength.to(u.nm),
            nonlinearity=nonlinearity,
            xy_stage=self.xy_stage,
            z_stage=self.z_stage,
            immersion_refractive_index=immersion_refractive_index,
            incident_field=incident_field,
            incident_transform=self._incident_transform,
            aberrations=aberrations,
            aberration_transform=aberration_transform,
        )

        super().__init__(source, self.psf, multi_threaded=multi_threaded)

        self._data_shape = output_shape
        self.pupil_field = self.psf.pupil_field  # detector that looks at the field in the pupil plane
        self.slm_aberration = self.psf.pupil_field._Pupil_Field  # detector that looks at aberrations and slm phase

    def _fetch(self, source: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Updates the image on the camera sensor.

        To compute the image:
        * First trigger the source, slm, and aberration sources
        * Then read the corresponding images.
        * Combines the slm and aberration images to compute the PSF
        * Crop the source image and upsample if needed
        * Convolve the source image with the PSF.
        * Compute the magnified and cropped image on the camera.

        Args:
            source: The source image (specimen) to be imaged.
            aberrations: The aberration pattern in the pupil plane.
            incident_field: The field from the SLM in the pupil plane.

        Returns:
            np.ndarray: The resulting image as it would appear on a camera sensor.
        """
        shift = Quantity((self.xy_stage.y, self.xy_stage.x))
        source = place(self.data_shape, self.pixel_size, source, shift)

        return fftconvolve(source, psf, mode="same")

    @property
    def abbe_limit(self) -> Quantity:
        """Returns the Abbe diffraction limit: λ/(2 NA).

        This is the theoretical resolution limit of the microscope due to diffraction.

        Returns:
            Quantity: The Abbe diffraction limit in length units.
        """
        return 0.5 * self.wavelength / self.numerical_aperture

    @property
    def numerical_aperture(self) -> float:
        return self.pupil_field.numerical_aperture

    @numerical_aperture.setter
    def numerical_aperture(self, value: float):
        self.pupil_field.numerical_aperture = value

    @property
    def wavelength(self) -> Quantity:
        return self.pupil_field.wavelength

    @wavelength.setter
    def wavelength(self, value: Quantity):
        value = value.to(u.nm)
        self.pupil_field.wavelength = value

    @property
    def nonlinearity(self) -> int:
        return self.psf.nonlinearity

    @nonlinearity.setter
    def nonlinearity(self, value: int):
        self.psf.nonlinearity = value

    @property
    def immersion_refractive_index(self) -> float:
        return self.pupil_field.immersion_refractive_index

    @immersion_refractive_index.setter
    def immersion_refractive_index(self, value: float):
        self.pupil_field.immersion_refractive_index = value

    @property
    def incident_transform(self) -> Optional[Transform]:
        """
        incident_transform:
        Optional Transform that transforms the phase pattern from the slm object
        (in slm.pixel_size units) to normalized pupil coordinates.
        Typically, the slm image is already in normalized pupil coordinates,
        but this transform can be used to mimic SLM misalignment.
        """
        return self._incident_transform

    @incident_transform.setter
    def incident_transform(self, value: Optional[Transform]):
        self.slm_aberration._incident_transform = value

    def z_stack_read(self, z: Quantitiy["length"]) -> np.ndarray:
        """Measures a z-stack by moving the z-stage to different positions and reading the corresponding images
        Args:
            z: Array of z positions to read at.

        Returns:
            Multidimensional array where imgs[iz,...] is the image at z position z[iz].
        """
        z_stack_images = np.zeros((len(z),) + self.data_shape)
        for ind, val in enumerate(z):
            self.z_stage.position = val
            z_stack_images[ind, ...] = self.read()
        return z_stack_images


class _Pupil_Field(Processor):
    def __init__(
        self,
        *,
        pupil_shape=None,
        pupil_extent=None,
        incident_field: Detector | None = None,
        incident_transform: Optional[Transform] = None,
        aberrations: Detector | None = None,
        aberration_transform: Optional[Transform] = None,
        multi_threaded: bool = True,
    ):

        super().__init__(aberrations, incident_field, multi_threaded=multi_threaded)
        self._pupil_shape = pupil_shape
        self._pupil_extent = pupil_extent
        self.aberration_transform = aberration_transform
        self._incident_transform = incident_transform

    def _fetch(
        self,
        aberrations: np.ndarray,  # noqa
        incident_field: np.ndarray,
    ) -> np.ndarray:

        # The aberrations and the SLM phase pattern are both mapped to the pupil plane coordinates
        pupil_field = patterns.disk(self._pupil_shape, radius=1.0, extent=self._pupil_extent)

        # Project aberrations
        if aberrations is not None:
            pupil_field = pupil_field * np.exp(
                1.0j
                * project(
                    aberrations,
                    source_extent=get_extent(aberrations),
                    out_extent=self._pupil_extent,
                    out_shape=self._pupil_shape,
                    transform=self.aberration_transform,
                    interp=cv2.INTER_LINEAR,
                )
            )

        # Project SLM fields
        if incident_field is not None:
            pupil_field = pupil_field * project(
                incident_field,
                out_extent=self._pupil_extent,
                out_shape=self._pupil_shape,
                transform=self._incident_transform,
            )
        return set_extent(pupil_field, self._pupil_extent)

    @property
    def data_shape(self) -> tuple:
        """Returns the shape of the image in the pupil plane.

        Returns:
            tuple: The dimensions of the output image (height, width).
        """
        return self._pupil_shape


class _Propagator(Processor):
    """
    Computes the field in the pupil plane of the microscope, given the SLM phase pattern and aberrations.
    The field is computed by multiplying the SLM phase pattern and aberrations and propagation due to z stage movement,
    and masking with the pupil function corresponding to the numerical aperture of the microscope objective.
    """

    def __init__(
        self,
        *,
        pupil_field: Detector | Processor,
        pupil_shape=None,
        pupil_extent=None,
        numerical_aperture: float = 1.0,
        wavelength: Quantity[u.nm],
        z_stage=None,
        immersion_refractive_index: Optional[float] = 1.0,
        multi_threaded: bool = True,
    ):
        self._pupil_field = pupil_field

        super().__init__(self._pupil_field, multi_threaded=multi_threaded)
        self._data_shape = pupil_shape
        self._pupil_extent = pupil_extent
        self.numerical_aperture = numerical_aperture
        self.wavelength = wavelength.to(u.nm)
        self.immersion_refractive_index = immersion_refractive_index
        self.z_stage = z_stage or LinearStage(0.1 * u.um)

    def _fetch(
        self,
        pupil_field: np.ndarray,
    ) -> np.ndarray:

        # Add defocus from z-stage
        if self.z_stage is not None:
            phase = propagation(
                self._data_shape,
                distance=self.z_stage.position,
                wavelength=self.wavelength,
                refractive_index=self.immersion_refractive_index,
                extent=self._pupil_extent,
                numerical_aperture=self.numerical_aperture,
            )
            pupil_field = pupil_field * np.exp(1j * phase)

        return set_extent(pupil_field, self._pupil_extent)

    @property
    def data_shape(self) -> tuple:
        return self._data_shape


class _PSF(Processor):
    def __init__(
        self,
        *,
        pupil_field: Detector | Processor,
        data_shape=None,
        pupil_extent=None,
        nonlinearity: int = 1,
        multi_threaded: bool = True,
    ):

        self.pupil_field = pupil_field

        super().__init__(self.pupil_field, multi_threaded=multi_threaded)
        self._data_shape = data_shape
        self._pupil_extent = pupil_extent
        self.nonlinearity = nonlinearity
        self._psf = None

    def _fetch(
        self,
        pupil_field: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the point spread function (PSF) of the microscope by performing a Fourier transform of the pupil field.

        Args:
            pupil_field: The field in the back focal plane of the microscope objective, which includes the effects of aberrations, SLM phase pattern.

        Returns:
            np.ndarray: The point spread function (PSF) of the microscope.
        """
        # pupil area for normalization of the PSF, so that a pupil fully filled with a field strength of 1.0 will produce an image
        # that has the same total intensity as the source image.
        pupil_area = patterns.disk(self.data_shape, radius=1.0, extent=self._pupil_extent)
        pupil_area = np.sum(pupil_area)  # TODO (efficiency): compute area directly from radius

        psf = np.abs(np.fft.ifft2(pupil_field)) ** 2

        psf = np.fft.ifftshift(psf) * (psf.size / pupil_area)
        # ifft_shift shifts psf by 1 pixel when off centre, both when the array is odd and even
        # Compensate for this by rolling the kernel by -1 pixel in both x and y directions
        psf = np.roll(psf, -1, axis=(0, 1))

        psf = psf**self.nonlinearity  # added for higher order microscopy (e.g. two-photon)
        return set_extent(psf, self._pupil_extent)

    @property
    def data_shape(self) -> tuple:
        return self._data_shape
