import numpy as np
from ..core import Detector, PhaseSLM
from .utilities import analyze_phase_stepping, WFSResult
from ..slm.patterns import tilt


class FourierBase:
    """Base class definition for the Fourier algorithms.

      Can run natively, provided you input the kspace for the reference and measurement part of the SLM.

      Attributes:
          feedback (Detector): Source of feedback
          slm (PhaseSLM): The spatial light modulator
          slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
            does not necessarily have to be the actual pixel dimensions as the SLM.
          phase_steps (int): The number of phase steps per mode. Default = 4
          overlap (float): A value between zero and one that indicates the fraction of overlap between the reference
            and measurement part of the SLM.
            A larger overlap reduces the uncertainty in matching the phase of the two halves of the solution,
            but reduces the overall efficiency of the algorithm. Default = 0.1

      [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
      "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
      """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape, k_left, k_right, phase_steps=4):
        """

        Args:
            feedback (Detector): The feedback source, usually a detector that provides measurement data.
            slm (PhaseSLM): slm object.
                The slm may have the `extent` property set to indicate the extent of the back pupil of the microscope
                objective in slm coordinates. By default, a value of 2.0, 2.0 is used (indicating that the pupil
                corresponds to a circle of radius 1.0 on the SLM). However, to prevent artefacts at the edges of the
                SLM,it may be overfilled, such that the `phases` image is mapped to an extent of e.g. (2.2, 2.2), i.e.
                10% larger than the back pupil.
            slm_shape (tuple[int, int]): The shape of the SLM patterns and transmission matrices.
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
        self._slm = slm
        self._feedback = feedback
        self.phase_steps = phase_steps
        self.k_left = k_left
        self.k_right = k_right
        self.slm_shape = slm_shape

    def execute(self) -> WFSResult:
        """
        Executes the FourierDualRef algorithm, computing the SLM transmission matrix.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        # left side experiment
        data_left = self.single_side_experiment(self.k_left, 0)

        # right side experiment
        data_right = self.single_side_experiment(self.k_right, 1)

        # Compute transmission matrix (=field at SLM), as well as noise statistics
        results = self.compute_t(data_left, data_right, self.k_left, self.k_right)
        results.left = data_left
        results.right = data_right
        results.k_left = self.k_left
        results.k_right = self.k_right
        return results

    def single_side_experiment(self, k_set, side):
        """
        Conducts experiments on one side of the SLM, generating measurements for each spatial frequency and phase step.

        Args:
            k_set (np.ndarray): An array of spatial frequencies to use in the experiment.
            side (int): Indicates which side of the SLM to use (0 for left, 1 for right).

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        measurements = np.zeros((k_set.shape[1], self.phase_steps, *self._feedback.data_shape))

        for i in range(k_set.shape[1]):
            for p in range(self.phase_steps):
                phase_offset = p * 2 * np.pi / self.phase_steps
                phase_pattern = self._get_phase_pattern(k_set[:, i], phase_offset, side)
                self._slm.set_phases(phase_pattern)
                self._feedback.trigger(out=measurements[i, p, ...])

        self._feedback.wait()
        return analyze_phase_stepping(measurements, axis=1)

    def _get_phase_pattern(self, k, phase_offset, side):
        """
        Generates a phase pattern for the SLM based on the given spatial frequency, phase offset, and side.

        Args:
            k (np.ndarray): A 2-element array representing the spatial frequency.
            phase_offset (float): The phase offset to apply to the pattern.
            side (int): Indicates the side of the SLM for the pattern (0 for left, 1 for right).

        Returns:
            np.ndarray: The generated phase pattern.
        """
        # tilt generates a pattern from -2.0 to 2.0 (The convention for Zernike modes normalized to an RMS of 1).
        # The natural step to take is the Abbe diffraction limit, which corresponds to a gradient from
        # -π to π.
        num_columns = int(0.5 * self.slm_shape[1])
        tilted_front = tilt([self.slm_shape[0], num_columns], [k[0] * (0.5 * np.pi), k[1] * (0.5 * np.pi)],
                            phase_offset=phase_offset, extent=self._slm.extent)

        # Handle side-dependent pattern

        empty_part = np.zeros((self.slm_shape[0], self.slm_shape[1] - num_columns))

        # Concatenate based on the side
        if side == 0:
            # Place the pattern on the left
            result = np.concatenate((tilted_front, empty_part), axis=1)
        else:
            # Place the pattern on the right
            result = np.concatenate((empty_part, tilted_front), axis=1)

        return result

    def compute_t(self, left: WFSResult, right: WFSResult, k_left, k_right) -> WFSResult:
        """
        Computes the SLM transmission matrix by combining the Fourier transmission matrices from both sides of the SLM.

        Args:
            left (WFSResult): The wavefront shaping result for the left side.
            right (WFSResult): The wavefront shaping result for the right side.
            k_left (np.ndarray): The spatial frequency matrix for the left side.
            k_right (np.ndarray): The spatial frequency matrix for the right side.

        Returns:
            WFSResult: An object containing the combined transmission matrix and related statistics.
        """

        # TODO: determine noise
        # Initialize transmission matrices
        t1 = np.zeros((*self.slm_shape, *self._feedback.data_shape), dtype='complex128')
        t2 = np.zeros((*self.slm_shape, *self._feedback.data_shape), dtype='complex128')

        # Calculate phase difference using the first mode

        # Find the index of the first mode in k_left and k_right
        index_first_mode_left = np.where((k_left == np.array([[0], [0]])).all(axis=0))[0][0]
        index_first_mode_right = np.where((k_right == np.array([[0], [0]])).all(axis=0))[0][0]

        phase_left_first_mode = np.angle(left.t[index_first_mode_left])
        phase_right_first_mode = np.angle(right.t[index_first_mode_right])

        # We can check here if they agree, because they should be equal
        phase_difference = (phase_left_first_mode - phase_right_first_mode) / 2

        # Apply phase correction to the right side
        phase_correction = np.exp(1j * phase_difference)

        # Construct the transmission matrices
        for n, t in enumerate(left.t):
            phi = self._get_phase_pattern(k_left[:, n], 0, 0)
            t1 += np.tensordot(np.exp(-1j * phi), t, 0)

        for n, t in enumerate(right.t):
            phi = self._get_phase_pattern(k_right[:, n], 0, 1)
            corrected_t = t * phase_correction  # Apply phase correction
            t2 += np.tensordot(np.exp(-1j * phi), corrected_t, 0)

        # Combine the left and right sides
        t_full = np.concatenate([t1[:, :self.slm_shape[0] // 2, ...], t2[:, self.slm_shape[0] // 2:, ...]], axis=1)

        # return combined result, along with a course estimate of the snr and expected enhancement
        # TODO: not accurate yet
        # for the estimated_improvement, first convert to field improvement, then back to intensity improvement
        def weighted_average(x_left, x_right):
            return (left.n * x_left + right.n * x_right) / (left.n + right.n)

        return WFSResult(t=t_full,
                         n=left.n + right.n,
                         axis=2,
                         noise_factor=weighted_average(left.noise_factor, right.noise_factor),
                         amplitude_factor=weighted_average(left.amplitude_factor, right.amplitude_factor),
                         non_linearity=weighted_average(left.non_linearity, right.non_linearity))
