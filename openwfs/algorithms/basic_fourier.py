from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .utilities import analyze_phase_stepping
from .fourier import FourierBase
from ..core import Detector, PhaseSLM


def build_square_k_space(k_min, k_max, k_step=1.0):
    """
    Constructs the k-space by creating a set of (k_x, k_y) coordinates.
    Fills the k_left and k_right matrices with the same k-space. (k_x, k_y) denote the k-space coordinates of the whole
    pupil. Only half SLM (and thus pupil) is modulated at a time, hence k_y (axis=1) must make steps of 2.

    Returns:
        k_space (np.ndarray): A 2xN array of k-space coordinates.
    """
    # Generate kx and ky coordinates
    kx_angles = np.arange(k_min, k_max + 1, 1)
    k_angles_min_even = (k_min if k_min % 2 == 0 else k_min + 1)        # Must be even
    ky_angles = np.arange(k_angles_min_even, k_max + 1, 2)              # Steps of 2

    # Combine kx and ky coordinates into pairs
    k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
    k_y = np.repeat(np.array(ky_angles)[:, np.newaxis], len(kx_angles), axis=1).flatten()
    k_space = np.vstack((k_x, k_y)) * k_step
    return k_space


class FourierDualReference(FourierBase):
    """Fourier double reference algorithm, based on Mastiani et al. [1].

    It constructs a square k-space coordinate grid for the algorithm. For custom k-spaces, you should use the
    FourierBase class. The k-space coordinates denote the entire pupil, not just one half. The k-space is normalized
    such that (1, 0) yields a -π to π gradient over the entire pupil.
    diffraction limit.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_angles_min: int = -3,
                 k_angles_max: int = 3, analyzer: Optional[callable] = analyze_phase_stepping):
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

        self._build_k_space()

    def _build_k_space(self):
        """
        Constructs the k-space by creating Cartesian products of k_x and k_y angles.
        Fills the k_left and k_right matrices with the same k-space. (k_x, k_y) denote the k-space coords of the whole
        SLM. Only half the SLM is modulated at a time, hence ky must make steps of 2.

        Returns:
            None: The function updates the instance attributes.
        """
        k_space = build_square_k_space(self.k_angles_min, self.k_angles_max)
        self.k_left = k_space
        self.k_right = k_space

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
        self._build_k_space()

    @property
    def k_angles_max(self) -> int:
        """The higher bound of the range of angles in x and y direction"""
        return self._k_angles_max

    @k_angles_max.setter
    def k_angles_max(self, value):
        """Sets the higher bound of the range of angles in x and y direction, triggers the building of the internal
            k-space properties."""
        self._k_angles_max = value
        self._build_k_space()


class FourierDualReferenceCircle(FourierBase):
    """
    Slightly altered version of Fourier double reference algorithm, based on Mastiani et al. [1].
    In this version, the k-space coordinates are restricted to lie within a circle of chosen radius.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """
    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, k_radius: float = 3.2,
                 k_step: float = 1.0, analyzer: Optional[callable] = analyze_phase_stepping):
        """
        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                does not necessarily have to be the actual pixel dimensions as the SLM.
            k_radius (float): Limit grid points to lie within a circle of this radius.
            k_step (float): Make steps in k-space of this value. 1 corresponds to diffraction limited tilt.
            phase_steps (int): The number of phase steps.
        """
        # TODO: Could be rewritten more compactly if we ditch the settable properties:
        # first build the k_space, then call super().__init__ with k_left=k_space, k_right=k_space.
        # TODO: Add custom grid spacing

        super().__init__(feedback=feedback, slm=slm, slm_shape=slm_shape, k_left=np.array((0, 0)),
                         k_right=np.array((0, 0)), phase_steps=phase_steps, analyzer=analyzer)

        self._k_radius = k_radius
        self.k_step = k_step
        self._build_k_space()

    def _build_k_space(self):
        """
        Constructs the k-space by creating Cartesian products of k_x and k_y angles.
        Fills the k_left and k_right matrices with the same k-space. (k_x, k_y) denote the k-space coordinates of the
        whole SLM. Only half the SLM is modulated at a time, hence k_y must make steps of 2.

        Returns:
            None: The function updates the instance attributes k_left and k_right.
        """
        k_radius = self.k_radius
        k_step = self.k_step
        k_max = int(np.floor(k_radius))
        k_space_square = build_square_k_space(-k_max, k_max, k_step=k_step)

        # Filter out k-space coordinates that are outside the circle of radius k_radius
        k_mask = (np.linalg.norm(k_space_square, axis=0) <= k_radius)
        k_space = k_space_square[:, k_mask]

        self.k_left = k_space
        self.k_right = k_space

    @property
    def k_radius(self) -> float:
        """The maximum radius of the k-space circle."""
        return self._k_radius

    @k_radius.setter
    def k_radius(self, value):
        """Sets the maximum radius of the k-space circle, triggers the building of the internal k-space properties."""
        self._k_radius = value
        self._build_k_space()

    def plot_k_space(self):
        """Plots the k-space coordinates."""
        phi = np.linspace(0, 2 * np.pi, 200)
        x = self.k_radius * np.cos(phi)
        y = self.k_radius * np.sin(phi)
        plt.plot(x, y, 'k')
        plt.plot(self.k_left[0, :], self.k_left[1, :], 'ob', label='k_left')
        plt.plot(self.k_right[0, :], self.k_right[1, :], '.r', label='k_right')
        plt.xlabel('k_x')
        plt.ylabel('k_y')
        plt.gca().set_aspect('equal')
