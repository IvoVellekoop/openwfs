import numpy as np
from ..core import Detector, PhaseSLM
from .utilities import analyze_phase_stepping, WFSResult


class StepwiseSequential:
    """
    Class definition for stepwise sequential algorithm for wavefront shaping, as described by Vellekoop [2].

    TODO: enable low-res pre-optimization (as in Vellekoop & Mosk 2007)
    TODO: modulate segments in pupil (circle) only

    [2]: Ivo M. Vellekoop, "Feedback-based wavefront shaping," Opt. Express 23, 12189-12206 (2015)
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, phase_steps=4, n_x=4, n_y=None):
        """
        This class systematically modifies the phase pattern of each SLM element and measures the resulting feedback.

        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            phase_steps (int): The number of phase steps.
            n_x (int): Number of SLM elements in x direction
            n_y (int): Number of SLM elements in y direction
        """
        self._n_x = n_x
        self._n_y = n_x if n_y is None else n_y
        self._slm = slm
        self._feedback = feedback
        self._phase_steps = phase_steps

    def execute(self) -> WFSResult:
        """
        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        phase_pattern = np.zeros((self.n_y, self.n_x), 'float32')
        measurements = np.zeros((self.n_y, self.n_x, self._phase_steps, *self._feedback.data_shape))

        for n_y in range(self.n_y):
            for n_x in range(self.n_x):
                for p in range(self._phase_steps):
                    phase_pattern[n_y, n_x] = p * 2 * np.pi / self._phase_steps
                    self._slm.set_phases(phase_pattern)
                    self._feedback.trigger(out=measurements[n_y, n_x, p, ...])
                phase_pattern[n_y, n_x] = 0

        self._feedback.wait()
        I_0 = np.mean(measurements[:, :, 0])  # flat wavefront intensity
        return analyze_phase_stepping(measurements, axis=2, A=np.sqrt(I_0))

    @property
    def n_x(self) -> int:
        """
        Returns:
            int: The number of SLM elements in the x direction.
        """
        return self._n_x

    @n_x.setter
    def n_x(self, value):
        self._n_x = value

    @property
    def n_y(self) -> int:
        """
        Returns:
            int: The number of SLM elements in the y direction.
        """
        return self._n_y

    @n_y.setter
    def n_y(self, value):
        self._n_y = value

    @property
    def phase_steps(self) -> int:
        """
        Returns:
            int: The number of phase steps used for each segment.
        """
        return self._phase_steps

    @phase_steps.setter
    def phase_steps(self, value):
        self._phase_steps = value
