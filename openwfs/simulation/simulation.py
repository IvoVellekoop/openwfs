import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
from typing import Annotated
from ..slm.patterns import gaussian
import astropy.units as u

class SimulatedWFS:
    """A Simulated 2D wavefront shaping experiment. Has a settable ideal wavefront, and can calculate a feedback
    image for a certain SLM pattern.

    It is both an SLM and a camera, so it can be loaded as both in the WFS algorithms for testing.

    Todo: the axis of the SLM & image plane are bogus, they should represent real values
    """

    def __init__(self, width=500, height=500, beam_profile_fwhm = None):
        """
        Initializer. Sets a flat illumination
        """
        self._beam_profile_fwhm = beam_profile_fwhm
        self.resized = True
        self.shape = (width, height)
        self.phases = np.zeros((width, height), dtype="float32")
        self.displayed_phases = 0
        self.exposure_ms = 1 * u.ms
        self._left = 0
        self._top = 0
        self._width = width
        self._height = height

        if beam_profile_fwhm is None:
            self.E_input_slm = np.ones((width, height), dtype="float32")
        else:
            self.E_input_slm = gaussian(width, fwhm=self.beam_profile_fwhm)

        self.ideal_wf = np.zeros((width, height), dtype="float32")
        self._image = None
        self.max_intensity = 1
        self.trigger()



    def trigger(self):
        """Triggers the virtual camera. This is where the intensity pattern on the camera is computed."""
        if self.resized:
            self._image = np.zeros((self.width, self.height), dtype=np.uint16)
            self.resized = False


        field_slm = self.E_input_slm * np.exp(1j * (self.displayed_phases - (self.ideal_wf)))
        field_slm_f = np.fft.fft2(field_slm)

        # scale image so that maximum intensity is 2 ** 16 - 1 for an input field of all 1
        scale_factor = np.sqrt(2**16 - 1) / np.prod(self.shape)
        image_plane = np.array((scale_factor * abs(np.fft.fftshift(field_slm_f))) ** 2)

        # the max intensity must be the highest found intensity, and at least 1.
        self.max_intensity = np.max([np.max(image_plane), self.max_intensity, 1]) # this is bad. It needs to have the same maximum
        self._image[:, :] = np.array((image_plane / self.max_intensity) * (2 ** 16 - 1), dtype=np.uint16)

    def read(self):
        return self._image

    def update(self, wait_factor=1.0, wait=False):
        """Update the phase on the virtual SLM. Note that the output only changes after triggering the 'camera'."""
        if np.any(np.array(self.phases.shape) > np.array(self.shape)):
            warnings.warn("Displayed wavefront is larger than simulation input and will be downscaled")
        self.displayed_phases = cv2.resize(self.phases, dsize=self.shape, interpolation=cv2.INTER_NEAREST)

    def wait(self):
        pass

    def reserve(self, time_ms):
        pass

    def set_data(self, pattern):
        """Now in radians"""
        pattern = np.array(pattern, dtype="float32", ndmin=2)
        self.phases = pattern

    def set_activepatch(self, id):
        pass

    def set_rect(self, rect):
        pass

    def destroy(self):
        pass

    def set_ideal_wf(self, ideal_wf):
        self.ideal_wf = cv2.resize(ideal_wf, dsize=self.shape, interpolation=cv2.INTER_NEAREST)

    @property
    def measurement_time(self):
        return self.exposure_ms

    @property
    def data_shape(self):
        return self.height, self.width

    @property
    def left(self) -> int:
        return self._top

    @left.setter
    def left(self, value: int):
        self._top = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def width(self) -> Annotated[int, {'min': 1, 'max': 1200}]:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value
        self._resized = True

    @property
    def height(self) -> Annotated[int, {'min': 1, 'max': 960}]:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value
        self._resized = True
    @property
    def Binning(self) -> int:
        return 1

    @property
    def beam_profile_fwhm(self) -> float:
        return self._beam_profile_fwhm

    @beam_profile_fwhm.setter
    def beam_profile_fwhm(self, value: float):
        self._beam_profile_fwhm = value
