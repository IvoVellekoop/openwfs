from .fourier import FourierBase
from ..core import Detector, PhaseSLM
import numpy as np


class FourierDualReference_new(FourierBase):
    """
    Implementation of the FourierDualRef algorithm with dynamic k-space based on overlap and number of modes.
    Ensures that the number of modes is always an odd number of integers and centered around 0.
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_space_radius = 2):
        """
        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for.
            phase_steps (int): The number of phase steps.
            overlap (float): The overlap value.
            number_modes (int): Total number of modes to be used for generating k-space.
        """
        super().__init__(feedback, slm, slm_shape, np.array((0, 0)), np.array((0, 0)), phase_steps=phase_steps)


        self._build_kspace(radius = k_space_radius)

    def _build_kspace(self, radius):
        """
        Constructs the k-space by creating a circular shape based on the specified radius.
        """
        # Define the maximum x and y indices based on the radius
        max_x = int(radius)  # Since we divide x by 2 in the condition
        max_y = int(radius/2)

        # Initialize lists to hold the kx and ky values
        k_x = []
        k_y = []

        # Iterate over possible x and y values to find those that meet the condition
        for x in range(-max_x, max_x + 1):
            for y in range(-max_y, max_y + 1):
                if np.sqrt((x) ** 2 + (y*2) ** 2) <= radius:
                    k_x.append(x)
                    k_y.append(y)

        # Convert the lists to numpy arrays
        k_x = np.array(k_x)
        k_y = np.array(k_y)

        # Assign the values to the class attributes
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
