import numpy as np

from .utilities import analyze_phase_stepping, WFSResult, DummyProgressBar
from ..core import Detector, PhaseSLM


class StepwiseSequential:
    """
    Class definition for stepwise sequential algorithm for wavefront shaping, as described by Vellekoop [1] [2].

    TODO: enable low-res pre-optimization (as in Vellekoop & Mosk 2007)
    TODO: modulate segments in pupil (circle) only

    [1]: Vellekoop, I. M., & Mosk, A. P. (2007). Focusing coherent light through opaque strongly scattering media.
    Optics Letters, 32(16), 2309-2311.
    [2]: Ivo M. Vellekoop, "Feedback-based wavefront shaping," Opt. Express 23, 12189-12206 (2015)
    """

    def __init__(
        self,
        feedback: Detector,
        slm: PhaseSLM,
        phase_steps: int = 4,
        n_x: int = 4,
        n_y: int = None,
    ):
        """
        This algorithm systematically modifies the phase pattern of each SLM element and measures the resulting
        feedback.

        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            phase_steps (int): The number of phase steps.
            n_x (int): Number of SLM elements in x direction
            n_y (int): Number of SLM elements in y direction
        """
        self._n_x = n_x
        self._n_y = n_x if n_y is None else n_y
        self._phase_steps = phase_steps
        self.slm = slm
        self.feedback = feedback

    def execute(self, progress_bar=DummyProgressBar()) -> WFSResult:
        """Executes the StepwiseSequential algorithm, computing the transmission matrix of the sample

        Returns:
            WFSResult: An object containing the computed transmission matrix and statistics.
        """
        phase_pattern = np.zeros((self.n_y, self.n_x), "float32")
        measurements = np.zeros((self.n_y, self.n_x, self.phase_steps, *self.feedback.data_shape))
        progress_bar.count = self.n_x * self.n_y

        for y in range(self.n_y):
            for x in range(self.n_x):
                for p in range(self.phase_steps):
                    phase_pattern[y, x] = p * 2 * np.pi / self.phase_steps
                    self.slm.set_phases(phase_pattern)
                    self.feedback.trigger(out=measurements[y, x, p, ...])
                phase_pattern[y, x] = 0
                progress_bar.update()

        self.feedback.wait()
        return analyze_phase_stepping(measurements, axis=2)

    @property
    def n_x(self) -> int:
        """The number of SLM elements in the x direction."""
        return self._n_x

    @n_x.setter
    def n_x(self, value):
        self._n_x = value

    @property
    def n_y(self) -> int:
        """The number of SLM elements in the y direction."""
        return self._n_y

    @n_y.setter
    def n_y(self, value):
        self._n_y = value

    @property
    def phase_steps(self) -> int:
        """The number of phase steps used for each segment."""
        return self._phase_steps

    @phase_steps.setter
    def phase_steps(self, value):
        self._phase_steps = value
