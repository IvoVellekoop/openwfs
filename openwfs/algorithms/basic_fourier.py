from .fourier import FourierDualRef
from ..core import DataSource, PhaseSLM
import numpy as np

class BasicFDR(FourierDualRef):
    """The most simple implementation of the FourierDualRef algorithm. It constructs a symmetric k-space for the algorithm.
    The k-space initializer is set to None because for custom k-spaces, you should use FourierDualRef directly.

    Attributes:
        k_x (numpy.ndarray): Array of k_x angles.
        k_y (numpy.ndarray): Array of k_y angles.
        k_left (numpy.ndarray): Left k-space matrix.
        k_right (numpy.ndarray): Right k-space matrix.

    Properties:
        k_angles_min (int): The minimum k-angle.
        k_angles_max (int): The maximum k-angle.

    Methods:
        build_kspace(): Constructs the k-space arrays used in the algorithm.
    """

    def __init__(self,feedback: DataSource, slm: PhaseSLM, slm_shape=(500,500), phase_steps=4, k_angles_min=-3, k_angles_max=3, overlap=0.1):
        """

        Args:
            feedback (DataSource): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                            does not necessarily have to be the actual pixel dimensions as the SLM.
            phase_steps (int): The number of phase steps.
            k_angles_min (int): The minimum k-angle.
            k_angles_max (int): The maximum k-angle.
            overlap (float): The overlap value.
        """
        super().__init__(feedback,slm, slm_shape, None, None, phase_steps=phase_steps, overlap=overlap)
        self._k_angles_min = k_angles_min
        self._k_angles_max = k_angles_max

        self.build_kspace()

    def build_kspace(self):
        """Constructs the k-space by creating Cartesian products of k_x and k_y angles.
        Filles the k_left and k_right matrices with the same k-space.

        Returns:
            None: The function updates the instance attributes.
        """
        kx_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        ky_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        # Make  the carthesian product of kx_angles and ky_angles to make a square kspace

        self.k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
        self.k_y = np.repeat(np.array(kx_angles)[:, np.newaxis], len(ky_angles), axis=1).flatten()
        self.k_left = np.vstack((self.k_x, self.k_y))
        self.k_right = np.vstack((self.k_x, self.k_y))

    @property
    def k_angles_min(self) -> int:
        return self._k_angles_min

    @k_angles_min.setter
    def k_angles_min(self, value):
        self._k_angles_min = value
        self.build_kspace()

    @property
    def k_angles_max(self) -> int:
        return self._k_angles_max

    @k_angles_max.setter
    def k_angles_max(self, value):
        self._k_angles_max = value
        self.build_kspace()