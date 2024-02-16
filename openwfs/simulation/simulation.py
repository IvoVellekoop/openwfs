import numpy as np
from typing import Optional
from ..utilities.patterns import gaussian
from .mockdevices import MockSLM, Processor


class SimulatedWFS(Processor):
    """A simple simulation of a wavefront shaping experiment.

    Simulates a configuration where both the SLM and the aberrations are placed in the back pupil
    of a microscope objective, and a point detector is placed in the the center of the focal plane.

    The simulation computes (Σ A·exp(i·(aberrations-slm)))², which is the 0,0 component of the Fourier transform
    of the field in the pupil plane (including aberrations and the correction by the SLM).

    For a more advanced (but slower) simulation, use `Microscope`
    """

    def __init__(self, aberrations: np.ndarray, beam_profile_waist=None):
        """
        Initializes the optical system with specified aberrations and optionally a Gaussian beam profile.

        This constructor sets up an optical system by specifying the aberrations across the SLM (Spatial Light
        Modulator) and, optionally, by defining a Gaussian beam profile. The aberrations array defines phase shifts at
        each point of the SLM. If a beam profile waist is provided, the input electric field is shaped into a
        Gaussian beam.

        Args:
            aberrations (np.ndarray): An array containing the aberrations in radians.

            beam_profile_waist (float, optional): The waist size of the Gaussian beam profile. If provided, the electric
            field at the SLM is shaped into a Gaussian beam with this waist size.

        The constructor creates a MockSLM instance based on the shape of the aberrations, calculates the electric
        field at the SLM considering the aberrations and optionally the Gaussian beam profile, and initializes the
        system with these parameters.
        """
        self.slm = MockSLM(aberrations.shape[0:2])
        self.E_input_slm = np.exp(1.0j * aberrations)  # electric field incident at the SLM
        if beam_profile_waist is not None:
            self.E_input_slm *= gaussian(aberrations.shape, waist=beam_profile_waist)

        # normalize the field
        self.E_input_slm *= 1 / np.linalg.norm(self.E_input_slm.ravel())
        super().__init__(self.slm.get_monitor('field'))

    def _fetch(self, slm_fields):  # noqa
        """
        Computes the intensity in the focus by applying phase corrections to the input electric field.

        This method adjusts the phase of the input electric field using the SLM (Spatial Light Modulator) phases,
        computes the tensor product with the electric field at the SLM, and then calculates the intensity of the
        resulting field.

        Args:
            out (Optional[np.ndarray]): An optional numpy array to store the calculated intensity. If provided, the
            intensity is stored in this array.

            slm_phases (np.ndarray): The phase corrections to apply, typically provided as a numpy array representing
            the phases set on the SLM.

        Returns:
            np.ndarray: A numpy array containing the calculated intensity in the focus.

        """
        field = np.tensordot(slm_fields, self.E_input_slm, 2)
        return np.abs(field) ** 2

    @property
    def data_shape(self):
        return self.E_input_slm.shape[2:]
