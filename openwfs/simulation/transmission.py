from typing import Optional

import numpy as np

from .slm import SLM, Processor
from ..utilities.patterns import ScalarType


class SimulatedWFS(Processor):
    """A simple simulation of a wavefront shaping experiment.

    Simulates a configuration where both the SLM and the aberrations are placed in the back pupil
    of a microscope objective, and a point detector is placed in the center of the focal plane.

    The simulation computes (Σ A·exp(i·(aberrations-slm)))², which is the 0,0 component of the Fourier transform
    of the field in the pupil plane (including aberrations and the correction by the SLM).

    For a more advanced (but slower) simulation, use `Microscope`
    """

    def __init__(
        self,
        *,
        t: Optional[np.ndarray] = None,
        aberrations: Optional[np.ndarray] = None,
        slm=None,
        multi_threaded=True,
        beam_amplitude: ScalarType = 1.0
    ):
        """
        Initializes the optical system with specified aberrations and optionally a Gaussian beam profile.

        This constructor sets up an optical system by specifying the aberrations across the SLM (Spatial Light
        Modulator) and, optionally, by defining a Gaussian beam profile. The aberrations array defines phase shifts at
        each point of the SLM. If a beam profile waist is provided, the input electric field is shaped into a
        Gaussian beam.

        Args:
            t: Transmission matrix. Must have the form (*feedback_shape, height, width), where feedback_shape
                is the shape of the feedback signal and may be 0 or more dimensional.
            aberrations: An array containing the aberrations in radians. Can be used instead of a transmission matrix,
                equivalent to specifying ``t = np.exp(1j * aberrations) / (aberrations.shape[0] * aberrations.shape[1])``.
            slm:
            multi_threaded (bool, optional): If True, the simulation will use multiple threads to compute the
                intensity in the focus. If False, the simulation will use a single thread. Defaults to True.
            beam_amplitude (ScalarType, optional): The amplitude profile of the incident beam. Defaults to 1.0.

        The constructor creates a MockSLM instance based on the shape of the aberrations, calculates the electric
        field at the SLM considering the aberrations and optionally the Gaussian beam profile, and initializes the
        system with these parameters.
        """

        # transmission matrix (normalized so that the maximum transmission is 1)
        self.t = t if t is not None else np.exp(1.0j * aberrations) / (aberrations.shape[0] * aberrations.shape[1])
        self.slm = slm if slm is not None else SLM(self.t.shape[-2:])

        super().__init__(self.slm.field, multi_threaded=multi_threaded)
        self.beam_amplitude = beam_amplitude

    def _fetch(self, incident_field):  # noqa
        """
        Computes the intensity in the focus by applying phase corrections to the input electric field.

        This method adjusts the phase of the input electric field using the SLM (Spatial Light Modulator) phases,
        computes the tensor product with the electric field at the SLM, and then calculates the intensity of the
        resulting field.

        Args:
            incident_field: The incident field (complex)

        Returns:
            np.ndarray: A numpy array containing the calculated intensity in the focus.

        """
        field = np.tensordot(self.t, incident_field * self.beam_amplitude, 2)
        return np.abs(field) ** 2

    @property
    def data_shape(self):
        return self.t.shape[:-2]
