import numpy as np
from typing import Union
from ..slm.patterns import gaussian
import astropy.units as u
from .mockdevices import MockSLM, Processor


class SimulatedWFS(Processor):
    """A simple simulation of a wavefront shaping experiment.

    Simulates a configuration where both the SLM and the aberrations are placed in the back pupil
    of a microscope objective, and a point detector is placed in the the center of the focal plane.

    The simulation computes (Σ A·exp(i·(aberrations-slm)))², which is the 0,0 component of the Fourier transform
    of the field in the pupil plane (including aberrations and the correction by the SLM).

    For a more advanced (but slower) simulation, use `Microscope`
    """

    def __init__(self, aberrations, beam_profile_waist=None):
        """
        Args:
            aberrations: array containing the aberrations in radians.
                The first two dimensions represent the y-x coordinates in the slm pattern.
                If `aberrations` has more than two dimensions, the feedback signal will be multi-dimensional
                and have the dimension `aberrations.shape[2:]
            beam_profile_waist:
        """
        slm = MockSLM(aberrations.shape[0:2])
        super().__init__(slm.pixels(), data_shape=aberrations.shape[2:], pixel_size=1 * u.dimensionless_unscaled)
        self.E_input_slm = np.exp(1.0j * aberrations)
        if beam_profile_waist is not None:
            self.E_input_slm *= gaussian(aberrations.shape, waist=beam_profile_waist)

        self.slm = slm

    def _fetch(self, out: Union[np.ndarray, None], slm_phases):
        """This is where the intensity in the focus is computed."""
        correction = np.exp(1.0j * slm_phases)
        field = np.tensordot(correction, self.E_input_slm, 2)
        intensity = abs(field) ** 2
        if out is not None:
            out[...] = intensity
        return intensity
