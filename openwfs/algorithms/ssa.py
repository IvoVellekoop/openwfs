import math
import numpy as np
from typing import Any, Annotated


class StepwiseSequential:
    """
    Class definition for stepwise sequential algorithm.
    (New approach)
    """

    def __init__(self, phase_steps=4, n_x=4, n_y=4, controller=None):
        self._n_x = n_x
        self._n_y = n_y
        self._controller = controller
        self._phase_steps = phase_steps

    def execute(self):
        self.controller.slm.phases = np.zeros((self.n_x, self.n_y), dtype="float32")
        self.controller.reserve((self.n_x, self.n_y, self.phase_steps))  # reserve space to hold the measurements

        phases = np.arange(self.phase_steps) / self.phase_steps * 2 * math.pi
        for n in range(self.n_x * self.n_y):
            for p in phases:
                self.controller.slm.phases.flat[n] = p
                self.controller.measure()

            self.controller.slm.phases.flat[n] = 0

        t = np.conj(self.controller.compute_transmission(self.phase_steps))
        return t[:,:,0]

    @property
    def n_x(self) -> int:
        return self._n_x

    @n_x.setter
    def n_x(self, value):
        self._n_x = value

    @property
    def n_y(self) -> int:
        return self._n_y

    @n_y.setter
    def n_y(self, value):
        self._n_y = value

    @property
    def phase_steps(self) -> int:
        return self._phase_steps

    @phase_steps.setter
    def phase_steps(self, value):
        self._phase_steps = value

    @property
    def controller(self) -> Any:
        return self._controller

