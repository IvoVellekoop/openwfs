.. _root-label:

OpenWFS
=====================================

..
    NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY.

.. only::html
    .. image:: https://readthedocs.org/projects/openwfs/badge/?version=latest
       :target: https://openwfs.readthedocs.io/en/latest/?badge=latest
       :alt: Documentation Status

What is wavefront shaping?
--------------------------------

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials. In WFS, a spatial light modulator (SLM) is used to spatially shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through :cite:`Vellekoop2007`, or inside :cite:`vellekoop2008demixing` a scattering material in such a way that the light interferes constructively at the desired focus; or light can be shaped to have other desired properties, such an optimal sensitivity for measurements :cite:`bouchet2021maximum`, specialized point-spread functions :cite:`boniface2017transmission` or for functions like optical trapping :cite:`vcivzmar2010situ`.

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for sample movement :cite:`valzania2023online`, to be optimally resilient to noise :cite:`mastiani2021noise`, or to use digital twin models to compute the required correction patterns :cite:`salter2014exploring,ploschner2015seeing,Thendiyammal2020,cox2023model`. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as using algorithms based on Hadamard pattern, or Fourier-based approaches :cite:`Mastiani2022`, Fast techniques that enable wavefront shaping in dynamic samples :cite:`Liu2017` :cite:`Tzang2019`, and many potential applications have been developed and prototyped, including endoscopy, optical trapping :cite:`Cizmar2010` and deep-tissue imaging :cite:`Streich2021`.

With the development of these advanced algorithms, however, the  complexity of WFS software is gradually is becoming a bottleneck for further advancements in the field, as well as for end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

What is OpenWFS?
----------------------

OpenWFS is a Python package for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:

    * **Spatial light modulator**. The SLM :class:`~.slm.SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated by OpenGL.
    * **Scanning microscope**. The :class:`~.devices.ScanningMicroscope` object uses National Instruments Data Acquisition Cards to control a laser-scanning microscope.
    * **GeniCam cameras**. The :class:`~.devices.Camera` object uses the `harvesters` backend to access any camera supporting the GeniCam standard.
    * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). See :ref:`synchronization`. The automatic synchronization makes it trivial to perform pipelined measurements that avoid the delay normally caused by the latency of the video card and SLM.

* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, most algorithms can be implemented in just a few lines of code without the need for low-level or hardware-specific programming.

* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

* **Platform for exchange and joint collaboration**. OpenWFS is designed to be a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools.

* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

* **MicroManager integration** (work in progress).  This code is designed so that it can be used in conjunction with `Micro-manager <https://micro-manager.org/>`_ :cite:`MMoverview`, a free and open-source microscopy software package, without any modification. To use this code in MicroManager, you need the PyDevice plugin, which can be found `here <https://www.github.com/IvoVellekoop/pydevice>`_ :cite:`pydevice`.

.. only:: latex

    Here, we first show how to get started using OpenWFS for simulating and controlling wavefront shaping experiments. An in-depth discussion of the core design of OpenWFS is given in Section :ref:`Key concepts`. The ability to simulate optical experiments is a key aspect of the package, which will be discussed in Section :ref:`Simulations`. Finally, OpenWFS is designed to be modular and easy to extend.  In Section :ref:`OpenWFS Development`, we show how to write custom hardware control modules and wavefront shaping algorithms.


Getting started
----------------------
OpenWFS is available on the PyPi repository, and the latest documentation can be found on `Read the Docs <https://openwfs.readthedocs.io/en/latest/>`_ :cite:`openwfsdocumentation`. To use OpenWFS, you need to have Python 3.9 or later installed. At the time of writing, OpenWFS is tested up to Python version 3.11 only since all dependencies were available for Python 3.12 yet. OpenWFS is developed and tested on Windows 11 and Ubuntu Linux.

To install OpenWFS, you can use pip:

.. code-block:: shell

    pip install openwfs

Below is a simple example that shows how to simulate a wavefront shaping experiment using OpenWFS. This example uses the `SimulatedWFS` class to simulate a spatial light modulator (SLM), and the propagation of light from that SLM, through a scattering medium, onto a detector.

.. code-block:: python

        import numpy as np
        from openwfs.simulation import SimulatedWFS
        from openwfs.algorithms import StepwiseSequential

        # Create a simple simulation of an experiment,
        # where light from an 'slm' is focused onto a 'detector'
        # through an aberrating plane with 25x25 segments.
        aberrations = np.random.uniform(0.0, 2 * np.pi, (25, 25))
        sim = SimulatedWFS(aberrations)
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

The code begins by importing the necessary modules and functions:.  Next, a `SimulatedWFS` object is created to simulate a basic wavefront experiment. The `slm` attribute of this object is a simulated spatial light modulator (SLM), which is used to shape the wavefront of the light. `SimulatedWFS` object itself acts as a detector that provides a feedback signal for wavefront shaping algorithms. This feedback signal corresponds to the light intensity in the focus of a microscope objective, with a specified aberration pattern applied to the pupil plane of that objective.

The `StepwiseSequential` :cite:`vellekoop2008phase` algorithm is then initialized. This algorithm is used to shape the wavefront in a way that maximizes the intensity of the light at the detector. As with all feedback-based wavefront shaping algorithms, this algorithm needs some feedback signal that it can use to optimize the wavefront., and it needs an SLM to control the wavefront. In this case, the feedback signal and the simulated SLM are both provided by the `SimulatedWFS` object. The algorithm returns the measured transmission matrix in the field `results.t`, which can be used to compute the optimal phase pattern to compensate the aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase pattern.

In summary, this code simulates a wavefront shaping experiment, uses the `StepwiseSequential` algorithm to optimize the phase pattern of the light, and measures the increase in intensity at the detector as a result of this optimization. The code to perform a real wavefront shaping experiment is similar, but it requires additional code to control the hardware.

.. code-block:: python

    import numpy as np
    from openwfs.slm import SLM
    from openwfs.devices import Camera
    from openwfs.processors import SingleRoi
    from openwfs.algorithms import StepwiseSequential

    slm = SLM(monitor=2)
    camera = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
    camera.exposure_time = 16.666 * u.ms
    camera.top = 400
    camera.left = 800
    camera.width = 10
    camera.height = 10
    feedback = SingleRoi(cam, mask_type='disk', radius=2.5)

    alg = StepwiseSequential(feedback=sim, slm=slm, n_x=10, n_y=10, phase_steps=4)

    # Run the algorithm
    result = alg.execute()

    # Measure intensity with flat and shaped wavefronts
    slm.set_phases(0)
    before = sim.read()
    slm.set_phases(-np.angle(result.t))
    after = sim.read()

    print(f"Wavefront shaping increased the intensity in the target from {before} to {after}")


As can be seen in the example, the code to control the hardware is similar to the code for the simulated experiment. This example assumes that a phase-only SLM is connected to the computer as a secondary monitor, and that a genicam-compatible camera was used to record the feedback signal. The `SingleRoi` object is used to define a region of interest over which the camera pixels are averaged to provide a feedback signal for the algorithm.


.. only:: html or markdown

    Bibliography
    --------------------
    .. bibliography::
