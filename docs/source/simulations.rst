.. _section-simulations:

Simulations
=======================
OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

Mock devices
+++++++++++++++++
Mock devices are primarily used for testing and debugging. These devices mimic the functionality of an actual device. OpenWFS currently includes the following mock devices:


*  :class:`~.StaticSource`, detector that always returns the same,  pre-set, data.

*  :class:`~.NoiseSource`, can be used as a drop-in replacement for any other detector object and produces random data.

* :class:`~.simulation.XYStage` simulates an x-y stage for which the position can be set and read.

* :class:`~.Shutter`, is a processor that either copies the data from the input detector, or returns a zero-filled array of the same size, mimicking a physical shutter that can be opened and closed.

* :class:`~ADCProcessor` is a processor that converts the data from the source to 16-bit unsigned integer values. In the process, it also mimics detector saturation, and optionally adds shot noise and readout noise. The latter is especially useful for predicting the effect of noise on a wavefront shaping algorithm.

* :class:`~Camera` is a processor that wraps a 2-D input source (such as a :class:`~StaticSource`) and adds functionality to set the region of interest through the `width`, `height`, `top` and `left` properties. It is a subclass of  :class:`~ADCProcessor`. It is particularly useful when interfacing with other code that expects unsigned integer data and a settable region of interest.

These classes play a pivotal role in automatic testing of various algorithms and experimental control scripts, as well as for testing the OpenWFS framework itself.

Simulating experiments
++++++++++++++++++++++++++++++++++
In addition to these classes, OpenWFS provides two means to simulate optical experiments. The first is the simple :class:`~SimulatedWFS` class, which simulates light propagation from an SLM through a scattering medium. The medium is represented by a transmission matrix `t`, which maps the field on the SLM to the field on the detector. The code below shows how to simulate a wavefront shaping algorithm using the `SimulatedWFS` class:

.. literalinclude:: ../../examples/wfs_simulated.py
    :language: python

The second method of simulating experiments is the :class:`~Microscope` class, which simulates a complete microscope, including aberrations and a pupil-conjugate SLM. The class is a :class:`~Processor`, that takes input from three sources: the specimen image, the SLM phase pattern, and the aberrations. When the microscope is 'triggered', it computes the point spread function using

.. math::

    \text{PSF} = \left\lVert\mathcal{F}\left( A_\text{SLM} e^{i (\phi_\text{aberrations} - \phi_\text{SLM})} \right) \right\rVert^2

where :math:`mathcal{F}` denotes a fast Fourier transform, :math:`\phi_\text{aberrations}` are the pupil-plane aberrations, and :math:`A_\text{SLM}` and :math:`\phi_\text{SLM}` are the amplitude and phase of the light that is imaged from the SLM to the pupil plane. This incident light is first cropped to the numerical aperture of the microscope, to realistically take into account the diffraction limit. Finally, the specimen image is convolved with the point spread function to return the microscope image.

Although this method does not simulate scattering in the sample, the approximation of fixed aberrations in the pupil plane of the microscope is valid for at least a small region in the sample, even in volumetric scattering tissue :cite:`osnabrugge2017generalized`.

In addition, the microscope has a simulated XY translation stage, allowing the simulation of a full motorized microscope, or testing the effect of sample drift on the wavefront shaping algorithm. Sample code for using the `Microscope` class can be found in the ``wfs_simulated_microscope.py`` script.


