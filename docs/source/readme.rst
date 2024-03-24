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

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials. In WFS, a spatial light modulator (SLM) is used to spatially shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through :cite:`Vellekoop2007`, or inside :cite:`vellekoop2008demixing` a scattering material in such a way that the light interferes constructively at the desired focus; or light can be shaped to have other desired properties, such an optimal sensitivity for specific measurements :cite:`bouchet2021maximum`, specialized point-spread functions :cite:`boniface2017transmission` or for functions like optical trapping :cite:`vcivzmar2010situ`.

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for sample movement :cite:`valzania2023online`, to be optimally resilient to noise :cite:`mastiani2021noise`, or to use digital twin models to compute the required correction patterns :cite:`salter2014exploring,ploschner2015seeing,Thendiyammal2020,cox2023model`. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as using algorithms based on Hadamard patterns, or Fourier-based approaches :cite:`Mastiani2022`, Fast techniques that enable wavefront shaping in dynamic samples :cite:`Liu2017,Tzang2019`, and many potential applications have been developed and prototyped, including endoscopy, optical trapping :cite:`Cizmar2010` and deep-tissue imaging :cite:`Streich2021`.

With the development of these advanced algorithms, however, the  complexity of WFS software is gradually is becoming a bottleneck for further advancements in the field, as well as for end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

What is OpenWFS?
----------------------

OpenWFS is a Python package for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:

    * **Spatial light modulator**. The :class:`~.slm.SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated by OpenGL.
    * **Scanning microscope**. The :class:`~.devices.ScanningMicroscope` object uses a National Instruments data Acquisition card to control a laser-scanning microscope.
    * **GeniCam cameras**. The :class:`~.devices.Camera` object uses the `harvesters` backend :cite:`harvesters` to access any camera supporting the GeniCam standard :cite:`genicam`.
    * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). The automatic synchronization makes it trivial to perform pipelined measurements that avoid the delay normally caused by the latency of the video card and SLM.

* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, most algorithms can be implemented in just a few lines of code without the need for low-level or hardware-specific programming.

* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

* **Platform for exchange and joint collaboration**. OpenWFS is designed to be a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools.

* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

.. only:: latex

    Here, we first show how to get started using OpenWFS for simulating and controlling wavefront shaping experiments. An in-depth discussion of the core design of OpenWFS is given in Section :numref:`Key concepts`. The ability to simulate optical experiments is a key aspect of the package, which will be discussed in Section :numref:`Simulations`. Finally, OpenWFS is designed to be modular and easy to extend.  In Section :numref:`OpenWFS Development`, we show how to write custom hardware control modules and wavefront shaping algorithms. Note that not all functionality of the package is covered in this document, and we refer to the API documentation :cite:`readthedocsOpenWFS`for a complete overview of most recent version of the package.


Getting started
----------------------
OpenWFS is available on the PyPI repository, and it can be installed with `pip install openwfs`. The latest documentation can be found on `Read the Docs <https://openwfs.readthedocs.io/en/latest/>`_ :cite:`openwfsdocumentation`. To use OpenWFS, you need to have Python 3.9 or later installed. At the time of writing, OpenWFS is tested up to Python version 3.11 only since not all dependencies were available for Python 3.12 yet. OpenWFS is developed and tested on Windows 11 and Ubuntu Linux.

Below is an example of how to use OpenWFS to run a simple wavefront shaping experiment.

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from openwfs.slm import SLM
    from openwfs.devices import Camera
    from openwfs.processors import SingleRoi
    from openwfs.algorithms import StepwiseSequential

    slm = SLM(monitor=2)
    camera = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
    camera.exposure_time = 16.666 * u.ms
    feedback = SingleRoi(cam, pos=(320, 320), mask_type='disk', radius=2.5)

    # Run the algorithm
    alg = StepwiseSequential(feedback=sim, slm=slm, n_x=10, n_y=10, phase_steps=4)
    result = alg.execute()

    # Measure intensity with flat and shaped wavefronts
    slm.set_phases(0)
    before = sim.read()
    slm.set_phases(-np.angle(result.t))
    after = sim.read()

    print(f"Wavefront shaping increased the intensity in the target from {before} to {after}")

This example illustrates several of the main concepts of OpenWFS. First, the code initializes an object to control a spatial light modulator (SLM) connected to a video port, and camera. The SLM is used to control the wavefront, and the camera is used to provide feedback to the wavefront shaping algorithm. The :class:`~.SingleRoi` object is a *processor* that takes images from the camera, and averages them over the specified circular region of interest.

Wavefront shaping is done using the `StepwiseSequential` :cite:`vellekoop2008phase` algorithm. The algorithm needs access to the SLM for controlling the wavefront, and gets feedback from the `SingleRoi` object. The algorithm returns the measured transmission matrix in the field `results.t`, which can be used to compute the optimal phase pattern to compensate the aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase pattern.

This code illustrates how OpenWFS separates the concerns of the hardware control (`SLM` and `Camera`), signal processing (`SingleROIProcessor`) and the algorithm itself (`StepwiseSequential`). A large variety of wavefront shaping experiments can be performed by using different types of feedback signals (such as optimizing multiple foci simultaneously using a :class:`~.MultiRoiProcessor` object), using different algorithms, or different image sources, such as a :class:`~.ScanningMicroscope`. Notably, these objects can be replaced by *mock* objects, that simulate the hardware and allow for rapid prototyping and testing of new algorithms without direct access to wavefront shaping hardware (see Section :numref:`Simulation`).


Analysis and Troubleshooting
----------------------
The principles of wavefront shaping are well established, and under close-to-ideal experimental conditions, it is possible to accurately predict the signal enhancement. In practice, however, there exist many practical issues that can negatively affect the outcome of the experiment.
OpenWFS has built-in functions to analyze and troubleshoot the measurements from a wavefront shaping experiment. These functions automatically estimate a number of different effects that can reduce the wavefront shaping fidelity.

The utility function `analyze_phase_stepping` not only extract the transmission matrix from the measurements, but also computes a series of troubleshooting statistics: it estimates the fidelity reduction factor due noise, unequal SLM illumination and incorrect phase calibration of the SLM.

The `troubleshoot` function computes several image frame metrics such as Contrast to Noise Ratio (CNR) and contrast enhancement. Furthermore, `troubleshoot` tests the image capturing repeatability and stability and estimates the fidelity reduction due to non-modulated light and decorrelation. Lastly, all fidelity reduction estimations are combined to make an order of magnitude estimation of the expected enhancement. `troubleshoot` returns an object containing the outcome of the different tests and analyses. The `troubleshoot` function can be used by replacing the `alg.execute()` line with for instance the following code:

.. code-block:: python

    # Run WFS troubleshooter and output a report to the console
    trouble = troubleshoot(algorithm=alg, background_feedback=roi_background, frame_source=cam, shutter=shutter)
    trouble.report()

In this example, `alg` is the wavefront shaping algorithm object, `roi_background` is a `SingleRoi` object that computes the average speckle intensity, `cam` is a `Camera` object and `shutter` is an object to control the shutter. The `report()` method prints a report of the analysis and test results to the console. For a comprehensive overview of the practical considerations in wavefront shaping, please see TODO: ref (gesubmit naar arxiv, zou maandag public moeten worden).

.. only:: html or markdown

    Bibliography
    --------------------
    .. bibliography::
