from typing import Optional

import numpy as np

from .utilities import analyze_phase_stepping
from .fourier import FourierBase
from ..core import Detector, PhaseSLM


class FourierDualReference(FourierBase):
    """Fourier double reference algorithm, based on Mastiani et al. [1].

    It constructs a square k-space coordinate grid for the algorithm. For custom k-spaces, you should use the
    FourierBase class. The k-space coordinates denote the entire pupil, not just one half. The k-space is normalized
    such that (0, 1) yields a -π to π gradient over the entire pupil.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_angles_min=-3,
                 k_angles_max=3, analyzer: Optional[callable] = analyze_phase_stepping):
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
        super().__init__(feedback, slm, slm_shape, np.array((0, 0)), np.array((0, 0)), phase_steps=phase_steps,
                         analyzer=analyzer)
        self._k_angles_min = k_angles_min
        self._k_angles_max = k_angles_max

        self._build_square_k_space()

    def _build_square_k_space(self):
        """
        Constructs the k-space by creating Cartesian products of k_x and k_y angles.
        Fills the k_left and k_right matrices with the same k-space. (k_x, k_y) denote the k-space coords of the whole
        SLM. Only half the SLM is modulated at a time, hence ky must make steps of 2.

        Returns:
            None: The function updates the instance attributes.
        """
        # Generate set of k-space coordinates
        kx_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        k_angles_min_even = (self._k_angles_min if self._k_angles_min % 2 == 0 else self._k_angles_min + 1)
        ky_angles = np.arange(k_angles_min_even, self._k_angles_max + 1, 2)

        k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
        k_y = np.repeat(np.array(ky_angles)[:, np.newaxis], len(kx_angles), axis=1).flatten()
        self.k_left = np.vstack((k_x, k_y))
        self.k_right = np.vstack((k_x, k_y))

    @property
    def k_angles_min(self) -> int:
        """The lower bound of the range of angles in x and y direction"""
        return self._k_angles_min

    @k_angles_min.setter
    def k_angles_min(self, value):
        """Sets the lower bound of the range of angles in x and y direction, triggers the building of the internal
            k-space properties.
        """
        self._k_angles_min = value
        self._build_square_k_space()

    @property
    def k_angles_max(self) -> int:
        """The higher bound of the range of angles in x and y direction"""
        return self._k_angles_max

    @k_angles_max.setter
    def k_angles_max(self, value):
        """Sets the higher bound of the range of angles in x and y direction, triggers the building of the internal
            k-space properties."""
        self._k_angles_max = value
        self._build_square_k_space()


class FourierDualReferenceCircle(FourierDualReference):
    """
    Slightly altered version of Fourier double reference algorithm, based on Mastiani et al. [1].
    In this version, the k-space coordinates are restricted to lie within a circle of chosen radius.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """
    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_radius_max=3,
                 analyzer: Optional[callable] = analyze_phase_stepping):
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
        k_angles_min = -k_radius_max
        k_angles_max = k_radius_max
        super().__init__(feedback=feedback, slm=slm, slm_shape=slm_shape, phase_steps=phase_steps,
                         k_angles_min=k_angles_min, k_angles_max=k_angles_max, analyzer=analyzer)

        # Filter out k-space coordinates that are outside the circle of radius k_radius_max
        k_left_mask = (np.linalg.norm(self.k_left, axis=0) <= k_radius_max)
        k_right_mask = (np.linalg.norm(self.k_right, axis=0) <= k_radius_max)
        self.k_left = self.k_left[:, k_left_mask]
        self.k_right = self.k_right[:, k_right_mask]
        self.k_radius_max = k_radius_max

        # TODO: remove k_min and k_max, by inheriting/overwriting directly

        import matplotlib.pyplot as plt
        phi = np.linspace(0, 2 * np.pi, 200)
        x = k_radius_max * np.cos(phi)
        y = k_radius_max * np.sin(phi)
        plt.plot(x, y, 'k')
        plt.plot(self.k_left[0, :], self.k_left[1, :], 'ob')
        plt.plot(self.k_right[0, :], self.k_right[1, :], '.r')
        plt.gca().set_aspect('equal')
        plt.show()
        pass
