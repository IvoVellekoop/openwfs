import numpy as np
from typing import Any, Annotated
from ..core import DataSource, PhaseSLM
from .utilities import analyze_phase_stepping
from ..slm.patterns import tilt


class FourierDualRef:
    """Base class definition for the Fourier algorithm as described by Mastiani et al. [1].

      Can run natively, provided you input the kspace for the reference and measurement part of the SLM.

      Attributes:
          feedback (DataSource): Source of feedback
          slm (PhaseSLM): The spatial light modulator
          slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                                        does not necessarily have to be the actual pixel dimensions as the SLM.
          phase_steps (int): The number of phase steps for the experiment.
          overlap (float): The overlap between the reference and measurement part of the SLM.
          t_slm (numpy.ndarray): The SLM transmission matrix.

      Returns:
          None

      [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
      "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
      """

    def __init__(self, feedback: DataSource, slm: PhaseSLM, slm_shape, k_left=None, k_right=None, phase_steps=4,
                 overlap=0.1):
        """

        Args:
          k_left (numpy.ndarray): The [k_x, k_y] matrix for the left side of shape (2, n).
          k_right (numpy.ndarray): The [k_x, k_y] matrix for the right side of shape (2, n).
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
        self.feedback_target = None

    def execute(self):
        """Execute the FourierDualRef algorithm. This computes the SLM transmission matrix.

        Returns:
            numpy.ndarray: The SLM transmission matrix.
        """
        # left side experiment
        measurements_left = self.single_side_experiment(self.k_left, 0)

        # the measurements are returned in a list of combinations of x&y angles, so only 1-dimensional
        self.t_left = analyze_phase_stepping(measurements_left, axis=1).field

        # right side experiment
        measurements_right = self.single_side_experiment(self.k_right, 1)
        self.t_right = analyze_phase_stepping(measurements_right, axis=1).field

        if len(self.t_left[0, ...].flatten()) == 1:  # If our feedback is 1 element
            self.feedback_target = [0]

        if self.feedback_target == None:
            print("Input a tuple in the shape of the feedback as feedback_target to compute the SLM transmission matrix"
                  "for that feedback point, then, execute compute_t to obtain the SLM transmission matrix")
            return None

        # calculate transmission matrix of the SLM plane from the Fourier transmission matrices:
        self.t_slm = self.compute_t(self.t_left, self.t_right, self.k_left,self.k_right)

        return self.t_slm

    def get_phase_pattern(self, k_x, k_y, phase_offset, side):
        height, width = self.slm_shape
        start = 0 if side == 0 else 0.5 - self._overlap / 2
        end = 0.5 + self._overlap / 2 if side == 0 else 1

        # Use tilt function
        tilted_front = tilt(width, (k_x, k_y))

        # Apply phase offset and handle side-dependent pattern
        final_pattern = np.zeros((width, height))
        num_columns = int((end - start) * height)

        if side == 0:
            final_pattern[:, :num_columns] = tilted_front[:, :num_columns] + phase_offset
        else:
            final_pattern[:, -num_columns:] = tilted_front[:, -num_columns:] + phase_offset

        return final_pattern

    def single_side_experiment(self, k_set, side):
        """
        Conducts the experiment on one side of the SLM and analyzes the result.

        Args:
            side (int): 0 for left, 1 for right side of the SLM.
        """
        measurements = np.zeros((len(k_set[0]), self.phase_steps, *self._feedback.data_shape))

        for i, (k_x, k_y) in enumerate(zip(k_set[0], k_set[1])):
            for p in range(self.phase_steps):
                phase_offset = p * 2 * np.pi / self.phase_steps
                phase_pattern = self.get_phase_pattern(k_x, k_y, phase_offset, side)
                self._slm.set_phases(phase_pattern)
                self._feedback.trigger(out=measurements[i, p, ...])
                self._feedback.wait()
            # Reset phase pattern if needed after each iteration

        # self._feedback.wait()
        return measurements

    def compute_t(self, t_fourier_left=None, t_fourier_right=None, k_left=None, k_right=None):
        """Computes the SLM transmission matrix from the Fourier transmission matrices.

        Args:
            t_fourier_left (numpy.ndarray): Fourier transmission matrix for the left side.
            t_fourier_right (numpy.ndarray): Fourier transmission matrix for the right side.
            k_left (numpy.ndarray): [k_x, k_y] matrix for the left side of shape (2, n).
            k_right (numpy.ndarray): [k_x, k_y] matrix for the right side of shape (2, n).

        Returns:
            numpy.ndarray: The SLM transmission matrix.
        """
        # Set defaults if None
        if t_fourier_left is None:
            t_fourier_left = self.t_left
        if t_fourier_right is None:
            t_fourier_right = self.t_right
        if k_left is None:
            k_left = self.k_left
        if k_right is None:
            k_right = self.k_right

        t_fourier_left = t_fourier_left[..., *self.feedback_target]
        t_fourier_right = t_fourier_right[..., *self.feedback_target]

        # bepaal ruis: bahareh. Find peak & dc ofset
        t1 = np.zeros((self.slm_shape[1], self.slm_shape[0]), dtype='complex128')
        t2 = np.zeros((self.slm_shape[1], self.slm_shape[0]), dtype='complex128')

        for n, t in enumerate(t_fourier_left):
            phi = self.get_phase_pattern(k_left[0, n], k_left[1, n], 0, 0)
            t1 += np.exp(1j * phi) * np.conj(t)

        for n, t in enumerate(t_fourier_right):
            phi = self.get_phase_pattern(k_right[0, n], k_right[1, n], 0, 1)
            t2 += np.exp(1j * phi) * np.conj(t)

        overlap_len = int(self._overlap * self.slm_shape[0])
        overlap_begin = self.slm_shape[0] // 2 - int(overlap_len / 2)
        overlap_end = self.slm_shape[0] // 2 + int(overlap_len / 2)

        if self._overlap != 0:
            c = np.vdot(t2[:, overlap_begin:overlap_end], t1[:, overlap_begin:overlap_end])
            factor = c / abs(c) * np.linalg.norm(t1[:, overlap_begin:overlap_end]) / np.linalg.norm(
                t2[:, overlap_begin:overlap_end])
            if np.linalg.norm(t2[:, overlap_begin:overlap_end]) == 0:
                factor = 1
            t2 = t2 * factor

            overlap = (t1[:, overlap_begin:overlap_end] + t2[:, overlap_begin:overlap_end]) / 2
            t_full = np.concatenate([t1[:, 0:overlap_begin], overlap, t2[:, overlap_end:]], axis=1)
        else:
            t_full = np.concatenate([t1[:, 0:overlap_begin], t2[:, overlap_end:]], axis=0)

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
