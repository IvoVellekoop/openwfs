import math
import numpy as np
from base_device_properties import *


class StepwiseSequential2:
    """
    Class definition for stepwise sequential algorithm.
    (New approach)
    """

    def __init__(self, **kwargs):
        parse_options(self, kwargs)

    def execute(self):
        self.slm.phases = np.zeros((self.N_x, self.N_y), dtype="float32")
        self.feedback.reserve((self.N_x, self.N_y, self.phase_steps))  # reserve space to hold the measurements

        phases = np.arange(self.phase_steps) / self.phase_steps * 2 * math.pi
        for n in range(self.N_x * self.N_y):
            for p in phases:
                self.slm.phases.flat[n] = p
                self.feedback.measure()

        t = np.tensordot(self.feedback.measurements, np.exp(-1j * phases),
                         ([2], [0]))  # perhaps include in feedback object as helper function?
        return t

    phase_steps = int_property(min=2, default=4, doc="Number of steps for the phase-stepping measurement. Defaults to "
                                                     "4 steps: 0, π/2, π, 3π/2")
    N_x = int_property(min=1, default=4, doc="Width of the wavefront texture, in segments")
    N_y = int_property(min=1, default=4, doc="Height of the wavefront texture, in segments")
    feedback = object_property()
    slm = object_property()
