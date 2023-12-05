import numpy as np
from ..core import Detector, PhaseSLM
from .utilities import analyze_phase_stepping
from ..utilities import imshow
from ..slm.patterns import tilt


class FourierDualRef:
    """Base class definition for the Fourier algorithm as described by Mastiani et al. [1].

      Can run natively, provided you input the kspace for the reference and measurement part of the SLM.

      Attributes:
          feedback (Detector): Source of feedback
          slm (PhaseSLM): The spatial light modulator
          slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                                        does not necessarily have to be the actual pixel dimensions as the SLM.
          phase_steps (int): The number of phase steps per mode.
            Default = 4
          overlap (float): A value between 0 and 1 that indicates the fraction of overlap between the reference
           and measurement part of the SLM.
           A larger overlap reduces the uncertainty in matching the phase of the two halves of the solution,
           but reduces the overall efficiency of the algorithm.
           Default = 0.1

      Returns:
          None

      [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
      "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
      """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape, k_left, k_right, phase_steps=4,
                 overlap=0.1):
        """

        Args:
          slm (PhaseSLM): slm object.
            The slm may have the `extent` property set to indicate the extent of the back pupil of the microscope
            objective in slm coordinates.
            By default, a value of 2.0,
            2.0 is used (indicating that the pupil corresponds to a circle of radius 1.0 on the SLM).
            However, to prevent artefacts at the edges of the SLM, it may be overfilled, such that the `phases` image
            is mapped to an extent of e.g. (2.2, 2.2), i.e. 10% larger than the back pupil.
          k_left (numpy.ndarray): 2-row matrix containing the y, and x components of the spatial frequencies
            used as basis for the left-hand side of the SLM.
            The frequencies are defined such that a frequency of (1,0) or (0,1) corresponds to
            a phase gradient of -π to π over the back pupil of the microscope objective, which results in
            a displacement in the focal plane of exactly a distance corresponding to the Abbe diffraction limit.
          k_right (numpy.ndarray): 2-row matrix containing the y and x components of the spatial frequencies
            for the right-hand side of the SLM.
            The number of frequencies need not be equal for k_left and k_right.
          phase_steps (int): The number of phase steps for each mode (default is 4).
          overlap (float): The overlap between the reference and measurement part of the SLM (default is 0.1).
        """
        self._execute_button = False
        self._phase_steps = phase_steps
        self._overlap = overlap
        self._slm = slm
        self._feedback = feedback
        self.k_left = k_left
        self.k_right = k_right
        self.slm_shape = slm_shape

    def execute(self):
        """Execute the FourierDualRef algorithm. This computes the SLM transmission matrix.

        Returns:
            numpy.ndarray: The SLM transmission matrix.
        """
        # left side experiment
        t_left = self.single_side_experiment(self.k_left, 0)

        # right side experiment
        t_right = self.single_side_experiment(self.k_right, 1)

        # calculate transmission matrix of the SLM plane from the Fourier transmission matrices:
        return self.compute_t(t_left, t_right, self.k_left, self.k_right)

    def get_phase_pattern(self, k_x, k_y, phase_offset, side):
        start = 0 if side == 0 else 0.5 - self._overlap / 2
        end = 0.5 + self._overlap / 2 if side == 0 else 1

        # tilt generates a pattern from -2 to 2 (The convention for Zernike modes normalized to an RMS of 1).
        # The natural step to take is the Abbe diffraction limit, which corresponds to a gradient from
        # -π to π.
        k_scale = 0.5 * np.pi
        tilted_front = tilt(self.slm_shape, (k_x * k_scale, k_y * k_scale), phase_offset=phase_offset,
                            extent=self._slm.extent)

        # Apply phase offset and handle side-dependent pattern
        final_pattern = np.zeros(self.slm_shape)
        num_columns = int((end - start) * self.slm_shape[1])

        if side == 0:
            final_pattern[:, :num_columns] = tilted_front[:, :num_columns]
        else:
            final_pattern[:, -num_columns:] = tilted_front[:, -num_columns:]

        return final_pattern

    def single_side_experiment(self, k_set, side):
        """
        Conducts the experiment on one side of the SLM and analyzes the result.

        Args:
            side (int): 0 for left, 1 for right side of the SLM.
        """
        measurements = np.zeros((k_set.shape[1], self.phase_steps, *self._feedback.data_shape))

        for i, (k_x, k_y) in enumerate(zip(k_set[0], k_set[1])):
            for p in range(self.phase_steps):
                phase_offset = p * 2 * np.pi / self.phase_steps
                phase_pattern = self.get_phase_pattern(k_x, k_y, phase_offset, side)
                self._slm.set_phases(phase_pattern)
                self._feedback.trigger(out=measurements[i, p, ...])

        self._feedback.wait()
        return analyze_phase_stepping(measurements, axis=1).field

    def compute_t(self, t_fourier_left, t_fourier_right, k_left, k_right):
        """Computes the SLM transmission matrix from the Fourier transmission matrices.

        Args:
            t_fourier_left (numpy.ndarray): Fourier transmission matrix for the left side.
            t_fourier_right (numpy.ndarray): Fourier transmission matrix for the right side.
            k_left (numpy.ndarray): [k_x, k_y] matrix for the left side of shape (2, n).
            k_right (numpy.ndarray): [k_x, k_y] matrix for the right side of shape (2, n).

        Returns:
            numpy.ndarray: The SLM transmission matrix.
        """

        # TODO: determine noise
        t1 = np.zeros((*self.slm_shape, *self._feedback.data_shape), dtype='complex128')
        t2 = np.zeros((*self.slm_shape, *self._feedback.data_shape), dtype='complex128')

        # TODO: why is there a - sign in the np.exp below?
        for n, t in enumerate(t_fourier_left):
            phi = self.get_phase_pattern(k_left[0, n], k_left[1, n], 0, 0)
            t1 += np.tensordot(np.exp(-1j * phi), t, 0)

        for n, t in enumerate(t_fourier_right):
            phi = self.get_phase_pattern(k_right[0, n], k_right[1, n], 0, 1)
            t2 += np.tensordot(np.exp(-1j * phi), t, 0)

        overlap_len = int(self._overlap * self.slm_shape[0])
        overlap_begin = self.slm_shape[0] // 2 - int(overlap_len / 2)
        overlap_end = self.slm_shape[0] // 2 + int(overlap_len / 2)

        c = np.sum(t2[:, overlap_begin:overlap_end, ...].conj() * t1[:, overlap_begin:overlap_end, ...], (0, 1))
        factor = c / abs(c) * np.linalg.norm(t1[:, overlap_begin:overlap_end]) / np.linalg.norm(
            t2[:, overlap_begin:overlap_end])
        if np.linalg.norm(t2[:, overlap_begin:overlap_end]) == 0:
            factor = 1
        t2 = t2 * factor

        overlap = 0.5 * (t1[:, overlap_begin:overlap_end, ...] + t2[:, overlap_begin:overlap_end, ...])
        t_full = np.concatenate([t1[:, 0:overlap_begin, ...], overlap, t2[:, overlap_end:, ...]], axis=1)

        return t_full

    @property
    def phase_steps(self) -> int:
        return self._phase_steps

    @phase_steps.setter
    def phase_steps(self, value):
        self._phase_steps = value

    @property
    def execute_button(self) -> bool:
        return self._execute_button

    @execute_button.setter
    def execute_button(self, value):
        self.execute()
        self._execute_button = value
