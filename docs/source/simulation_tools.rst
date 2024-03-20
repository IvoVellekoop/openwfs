Simulations
=======================



OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

TODO: introduction, explain Microscope, SimulatedWFS classes.
Explain ADCConverter, Camera classes (for adding noise).
Give example of a simulated WFS experiment.


.. code-block:: python

        import numpy as np
        from openwfs.simulation import SimulatedWFS
        from openwfs.algorithms import StepwiseSequential

        # Create a simple simulation of an experiment,
        # where light from an 'slm' is focused onto a 'detector'
        # through a random phase plate with 25x25 segments.
        sim = SimulatedWFS(aberrations=np.random.uniform(0.0, 2 * np.pi, (25, 25)))
        slm = sim.slm

        # Use the StepwiseSequential algorithm to optimize the phase pattern,
        # using a correction pattern of 10x10 segments and 4 phase steps
        alg = StepwiseSequential(feedback=sim, slm=slm, n_x=10, n_y=10, phase_steps=4)
        result = alg.execute()

        # Measure intensity with flat and shaped wavefronts
        slm.set_phases(0)
        before = sim.read()
        slm.set_phases(-np.angle(result.t))
        after = sim.read()

        print(f"Wavefront shaping increased the intensity in the target from {before} to {after}")
