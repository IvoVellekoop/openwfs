from .no_overlap_fourier_base import FourierBase2
from ..core import Detector, PhaseSLM
import numpy as np


class FourierDualReference_new(FourierBase2):
    """
    Implementation of the FourierDualRef algorithm with dynamic k-space based on overlap and number of modes.
    Ensures that the number of modes is always an odd number of integers and centered around 0.
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, number_modes=9):
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
        self._number_modes = max(number_modes, 9)  # Ensure a minimum of 9 modes

        self._build_kspace()

    def _build_kspace(self):
        """
        Constructs the k-space by creating a rectangle shape based on overlap and number of modes.

        ToDo: we now always make our k-space symmetric. We could forfeit that requirement and have a better fitting
            k-space. That would mean entering max modes = 12 would produce kx = [-1, 0, 1, 2] & ky = [-1, 0, 1].
            Whether to add positive modes or negative ones first is arbitrary.
        """
        ratio = 2 - self._overlap
        y_modes = int(np.round(np.sqrt(self._number_modes / ratio)))

        # Ensure y_modes is odd and at least 3
        y_modes = max(y_modes + (y_modes % 2 == 0), 3)
        x_modes = max(int(np.round(y_modes * ratio)) + (int(np.round(y_modes * ratio)) % 2 == 0), 3)

        # Adjust if total exceeds max allowed modes
        while x_modes * y_modes > self._number_modes:
            if x_modes > y_modes * ratio or y_modes <= 3:
                x_modes -= 2  # Reduce by 2 to keep it odd
            else:
                y_modes -= 2  # Reduce by 2 to keep it odd

        kx = np.linspace(-int((x_modes - 1) / 2), int((x_modes - 1) / 2), x_modes)
        ky = np.linspace(-int((y_modes - 1) / 2), int((y_modes - 1) / 2), y_modes)

        k_x = np.repeat(kx[np.newaxis, :], y_modes, axis=0).flatten()
        k_y = np.repeat(ky[:, np.newaxis], x_modes, axis=1).flatten()

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
