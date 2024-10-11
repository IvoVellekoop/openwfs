from typing import Optional

import numpy as np

from .dual_reference import DualReference
from .utilities import analyze_phase_stepping
from ..core import Detector, PhaseSLM
from ..utilities import tilt


class FourierDualReference(DualReference):
    """Fourier double reference algorithm, based on Mastiani et al. [1].

    Improvements over [1]:

    - The set of plane waves is taken from a disk in k-space instead of a square.
    - No overlap between the two halves is needed, instead the final stitching step is done
      using measurements already in the data set.
    - When only a single target is optimized, the algorithm can be used in an iterative version
      to increase SNR during the measurement, similar to [2].

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)

    [2]: X. Tao, T. Lam, B. Zhu, et al., “Three-dimensional focusing through scattering media using conjugate adaptive
    optics with remote focusing (CAORF),” Opt. Express 25, 10368–10383 (2017).
    """

    def __init__(
        self,
        *,
        feedback: Detector,
        slm: PhaseSLM,
        slm_shape=(500, 500),
        phase_steps=4,
        k_radius: float = 3.2,
        k_step: float = 1.0,
        iterations: int = 2,
        amplitude: np.ndarray = 1.0,
        optimized_reference: Optional[bool] = None
    ):
        """
        Args:
            feedback: Source of feedback
            slm: The spatial light modulator
            slm_shape: The shape that the SLM patterns & transmission matrices are calculated for,
                does not necessarily have to be the actual pixel dimensions as the SLM.
            phase_steps: The number of phase steps.
            k_radius: Limit grid points to lie within a circle of this radius.
            k_step: Make steps in k-space of this value. 1 corresponds to diffraction limited tilt.
            iterations: Number of ping-pong iterations. Defaults to 2.
            amplitude: Amplitude profile over the SLM. Defaults to 1.0 (flat)
            optimized_reference:
                When `True`, during each iteration the other half of the SLM displays the optimized pattern so far (as in [1]).
                When `False`, the algorithm optimizes A with a flat wavefront on B,
                and then optimizes B with a flat wavefront on A.
                This mode also allows for multi-target optimization.
                When set to `None` (default), the algorithm uses True if there is a single target,
                and False if there are multiple targets.


        """
        self._k_radius = k_radius
        self._k_step = k_step
        self._shape = slm_shape
        group_mask = np.zeros(slm_shape, dtype=bool)
        group_mask[:, slm_shape[1] // 2 :] = True
        super().__init__(
            feedback=feedback,
            slm=slm,
            phase_patterns=self._construct_modes(),
            group_mask=group_mask,
            phase_steps=phase_steps,
            iterations=iterations,
            amplitude=amplitude,
            optimized_reference=optimized_reference,
        )

    def _construct_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Constructs the set of plane wave modes."""

        # start with a grid of k-values
        # then filter out the ones that are outside the circle
        # in the grid, the spacing in the kx direction is twice the spacing in the ky direction
        # because we subdivide the SLM into two halves along the x direction,
        # which effectively doubles the number of kx values
        int_radius_x = np.ceil(self.k_radius / (self.k_step * 2))
        int_radius_y = np.ceil(self.k_radius / self.k_step)
        kx, ky = np.meshgrid(
            np.arange(-int_radius_x, int_radius_x + 1) * (self.k_step * 2),
            np.arange(-int_radius_y, int_radius_y + 1) * self.k_step,
        )

        # only keep the points within the circle
        mask = kx**2 + ky**2 <= self.k_radius**2
        k = np.stack((ky[mask], kx[mask])).T

        # construct the modes for these kx ky values
        modes = np.zeros((len(k), *self._shape), dtype=np.float32)
        for i, k_i in enumerate(k):
            # tilt generates a pattern from -2.0 to 2.0 (The convention for Zernike modes normalized to an RMS of 1).
            # The natural step to take is the Abbe diffraction limit of the modulated part,
            # which corresponds to a gradient from -π to π over the modulated part.
            # TODO: modify tilt to take a 2-D argument, returning the mode set directly?
            modes[i] = tilt(self._shape, g=k_i * 0.5 * np.pi)

        return modes, modes

    @property
    def k_radius(self) -> float:
        """The maximum radius of the k-space circle."""
        return self._k_radius

    @k_radius.setter
    def k_radius(self, value: float):
        """Sets the maximum radius of the k-space circle, triggers the building of the internal k-space properties."""
        self._k_radius = float(value)
        self.phase_patterns = self._construct_modes()

    @property
    def k_step(self) -> float:
        return self._k_step

    @k_step.setter
    def k_step(self, value: float):
        self._k_step = float(value)
        self.phase_patterns = self._construct_modes()
