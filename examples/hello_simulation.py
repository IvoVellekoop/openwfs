"""Hello, Simulation!
===============================================
Simulates a wavefront shaping experiment using a SimulatedWFS object,
which acts both as a spatial light modulator (SLM) and a detector."""
import numpy as np

from openwfs.algorithms import StepwiseSequential
from openwfs.simulation import SimulatedWFS

# Create a simulation of an experiment
size = (25, 25)
t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
sim = SimulatedWFS(t=t)
slm = sim.slm

# Use the StepwiseSequential algorithm to optimize the phase pattern,
# using a correction pattern of 10x10 segments and 4 phase steps
# The pattern is scaled to the size of the transmission matrix t automatically
alg = StepwiseSequential(feedback=sim, slm=slm, n_x=10, n_y=10, phase_steps=4)
result = alg.execute()

# Measure intensity with flat and shaped wavefronts
slm.set_phases(0)
before = sim.read()
slm.set_phases(-np.angle(result.t))
after = sim.read()
print(f"Intensity in the target increased from {before} to {after}")
