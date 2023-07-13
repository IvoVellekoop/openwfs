import math
import numpy as np
from base_device_properties import *

class StepwiseSequential:
    """
    Class definition for stepwise sequential algorithm.
    (New approach)
    """

    def __init__(self, **kwargs):
        parse_options(self, kwargs)

    def execute(self):
        self.slm.phases = np.zeros(self.Nx, self.Ny, np.float32)
        self.feedback.synchronize_with(self.slm)
        self.feedback.reserve((self.Nx, self.Ny, self.phase_steps))

        phases = np.range(self.phase_steps) / self.phase_steps * 2 * math.pi
        for n in range(self.Nx * self.Ny):
            for p in phases:
                self.slm.phases[n] = p
                self.feedback.measure()

        t = np.tensordot(self.feedback.measurements, np.exp(-1j * phases), ([2], [0]))
        return t

    phase_steps = int_property(min=2, default=4, doc="Number of steps for the phase-stepping measurement. Defaults to "
                                                     "4 steps: 0, π/2, π, 3π/2")
    n_x = int_property(min=1, default=4, doc="Width of the wavefront texture, in segments")
    n_y = int_property(min=1, default=4, doc="Height of the wavefront texture, in segments")
    feedback = object_property()
    slm = object_property()