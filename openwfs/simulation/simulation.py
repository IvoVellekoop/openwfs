import numpy as np
import cv2
from ..slm.patterns import gaussian
import astropy.units as u
from .mockdevices import MockSLM, Generator


class SimulatedWFS:
    """A Simulated 2D wavefront shaping experiment. Has a settable ideal wavefront, and can calculate a feedback
    image for a certain SLM pattern.

    It is both an SLM and a camera, so it can be loaded as both in the WFS algorithms for testing.

    Todo: the axis of the SLM & image plane are bogus, they should represent real values
    """

    def __init__(self, aberrations, width=500, height=500, beam_profile_waist=None):
        """
        Initializer. Sets a flat illumination
        """
        shape = (height, width)
        aberrations = cv2.resize(aberrations, dsize=shape, interpolation=cv2.INTER_NEAREST)
        self.E_input_slm = np.exp(1.0j * aberrations)
        if beam_profile_waist is not None:
            self.E_input_slm *= gaussian(width, waist=beam_profile_waist)

        self.slm = MockSLM(shape)
        self.cam = Generator(self.compute_image, data_shape=shape, pixel_size=10 * u.um)

    def compute_image(self, shape):
        """This is where the intensity pattern on the camera is computed."""
        assert shape == self.E_input_slm.shape
        field_slm = self.E_input_slm * np.exp(1.0j * self.slm.phases)
        field_slm_f = np.fft.fft2(field_slm)
        return abs(np.fft.fftshift(field_slm_f)) ** 2
