import astropy.units as u
import numpy as np

from openwfs.algorithms import StepwiseSequential
from openwfs.devices import SLM, Camera
from openwfs.processors import SingleRoi

# Display the SLM patterns on the secondary monitor
slm = SLM(monitor_id=2)

# Connect to a GenICam camera
camera = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
camera.exposure_time = 16.666 * u.ms
feedback = SingleRoi(camera, pos=(320, 320), mask_type='disk', radius=2.5)

# Run the algorithm
alg = StepwiseSequential(feedback=feedback, slm=slm, n_x=10, n_y=10, phase_steps=4)
result = alg.execute()

# Measure intensity with flat and shaped wavefronts
slm.set_phases(0)
before = feedback.read()
slm.set_phases(-np.angle(result.t))
after = feedback.read()

print(f"Wavefront shaping increased the intensity in the target from {before} to {after}")
