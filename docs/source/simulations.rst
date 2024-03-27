.. _section-simulations:

Simulations
=======================
OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

Mock devices
+++++++++++++++++
Mock devices are primarily used for testing and debugging. These devices mimic the functionality of an actual device. OpenWFS currently includes the following mock devices:


*  :class:`~.StaticSource`, detector that always returns the same,  pre-set, data.

*  :class:`~.NoiseSource`, can be used as a drop-in replacement for any other detector object and produces random data.

* :class:`~.simulation.XYStage` simulates an x-y stage that can be moved to a certain position.

* :class:`~.Shutter`, is a processor that either copies the data from the input detector, or returns a zero-filled array of the same size, mimicking a physical shutter that can be opened and closed.

* :class:`~ADCConverter` is a processor that converts the data from the source to 16-bit unsigned integer values. In the process, it also mimics detector saturation, and optionally adds shot noise and readout noise. The latter is especially useful when testing the effect of noise on a wavefront shaping algorithm.

* :class:`~Camera` is a processor that wraps a 2-D input source (such as a :class:`~StaticSource`) and adds functionality to set the region of interest through the `width`, `height`, `top` and `left` properties. It is a subclass of  :class:`~ADCConverter`. It is particularly useful when interfacing with other code that expects unsigned integer data and a settable region of interest.

These classes play a pivotal role in automatic testing of various algorithms and experimental control scripts, as well as for testing the OpenWFS framework itself.

Simulating experiments
++++++++++++++++++++++++++++++++++
In addition to these classes, OpenWFS provides two means to simulate optical experiments.

The first is the simple :class:`~SimulatedWFS` class, which simulates light propagation from an SLM through a scattering medium. The medium is represented by a transmission matrix `t`, which maps the field on the SLM to the field on the detector. If `t` is a 2-dimensional array, it corresponds to propagation from the 2-dimensional SLM to a point detector. If `t` is a multi-dimensional array, the last two dimensions correspond to the SLM, and the remaining dimensions correspond to the detector 'pixels'.

The code below shows how to simulate a wavefront shaping algorithm using the `SimulatedWFS` class:

.. code-block:: python

        import numpy as np
        from openwfs.simulation import SimulatedWFS
        from openwfs.algorithms import StepwiseSequential

        # Create a simple simulation of an experiment,
        # where light from an 'slm' is focused onto a 'detector'
        # through a random phase plate with 25x25 segments.
        t = exp(i * (np.random.uniform(0.0, 2 * np.pi, (25, 25))))
        sim = SimulatedWFS(t)
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


The second method of simulating experiments is the :class:`~Microscope` class, which simulates a complete microscope, including aberrations and a pupil-conjugate SLM. The class is a :class:`~Processor`, that takes input from three sources: the specimen image, the SLM phase pattern, and the aberrations. When the microscope is 'triggered', it computes the point spread function using

.. math::

    \text{PSF} = \left\lVert\mathcal{F}\left( A_\text{SLM} e^{i (\phi_\text{aberrations} - \phi_\text{SLM})} \right) \right\rVert^2

where :math:`mathcal{F}` denotes a fast Fourier transform, :math:`\phi_\text{aberrations}` are the pupil-plane aberrations, and :math:`A_\text{SLM}` and :math:`\phi_\text{SLM}` are the amplitude and phase of the light that is imaged from the SLM to the pupil plane. This incident light is first cropped to the numerical aperture of the microscope, to realistically take into account the diffraction limit. Finally, the specimen image is convolved with the point spread function to return the microscope image.

Although this method does not simulate scattering in the sample, the approximation of fixed aberrations in the pupil plane of the microscope is valid for at least a small region in the sample, even in volumetric scattering tissue :cite:`osnabrugge2017generalized`.

In addition, the microscope has a simulated XY translation stage, allowing the simulation of a full motorized microscope, or testing the effect of sample drift on the wavefront shaping algorithm. Sample code for using the `Microscope` class can be found in the ``wfs_simulated_microscope.py`` script.


