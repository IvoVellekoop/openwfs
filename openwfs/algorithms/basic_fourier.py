from .fourier import FourierBase
from ..core import Detector, PhaseSLM
import numpy as np


class FourierDualReference(FourierBase):
    """Fourier double reference algorithm, based on Mastiani et al. [1].

    It constructs a symmetric k-space for the algorithm.
    The k-space initializer is set to None because for custom k-spaces, you should use FourierDualRef directly.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_angles_min=-3,
                 k_angles_max=3):
        """
        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                does not necessarily have to be the actual pixel dimensions as the SLM.
            phase_steps (int): The number of phase steps.
            k_angles_min (int): The minimum k-angle.
            k_angles_max (int): The maximum k-angle.
        """
        super().__init__(feedback, slm, slm_shape, np.array((0, 0)), np.array((0, 0)), phase_steps=phase_steps)
        self._k_angles_min = k_angles_min
        self._k_angles_max = k_angles_max

        self._build_kspace()

    def _build_kspace(self):
        """
        Constructs the k-space by creating Cartesian products of k_x and k_y angles.
        Fills the k_left and k_right matrices with the same k-space.

        Returns:
            None: The function updates the instance attributes.
        """
        kx_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        ky_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        # Make the carthesian product of kx_angles and ky_angles to make a square kspace

        k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
        k_y = np.repeat(np.array(ky_angles)[:, np.newaxis], len(ky_angles), axis=1).flatten()
        self.k_left = np.vstack((k_x, k_y))
        self.k_right = np.vstack((k_x, k_y))

    @property
    def k_angles_min(self) -> int:
        """The lower bound of the range of angles in x and y direction"""
        return self._k_angles_min

    @k_angles_min.setter
    def k_angles_min(self, value):
        """Sets the lower bound of the range of angles in x and y direction, triggers the building of the internal
            kspace properties.
        """
        self._k_angles_min = value
        self._build_kspace()

    @property
    def k_angles_max(self) -> int:
        """The higher bound of the range of angles in x and y direction"""
        return self._k_angles_max

    @k_angles_max.setter
    def k_angles_max(self, value):
        """Sets the higher bound of the range of angles in x and y direction, triggers the building of the internal
            kspace properties."""
        self._k_angles_max = value
        self._build_kspace()
