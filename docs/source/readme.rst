.. _root-label:

OpenWFS
=====================================

..
    NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY.

.. only:: html

    .. image:: https://readthedocs.org/projects/openwfs/badge/?version=latest
       :target: https://openwfs.readthedocs.io/en/latest/?badge=latest
       :alt: Documentation Status

.. only:: markdown

   [![PyTest](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml)
   [![Black](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml)

What is wavefront shaping?

What is wavefront shaping?
--------------------------------

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials :cite:`kubby2019`. In WFS, a spatial light modulator (SLM) is used to shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through :cite:`Vellekoop2007`, or inside :cite:`vellekoop2008demixing` scattering materials; or light can be shaped to have other desired properties, such as optimal sensitivity for specific measurements :cite:`bouchet2021maximum`, specialized point-spread functions :cite:`boniface2017transmission` or for functions like optical trapping :cite:`vcivzmar2010situ`.

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for sample movement :cite:`valzania2023online`, to be optimally resilient to noise :cite:`mastiani2021noise`, or to use digital twin models to compute the required correction patterns :cite:`salter2014exploring,ploschner2015seeing,Thendiyammal2020,cox2023model`. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as using algorithms based on Hadamard patterns, or Fourier-based approaches :cite:`Mastiani2022`. Fast techniques that enable wavefront shaping in dynamic samples :cite:`Liu2017,Tzang2019`, and many potential applications have been developed and prototyped, including endoscopy :cite:`ploschner2015seeing`, optical trapping :cite:`Cizmar2010` and deep-tissue imaging :cite:`Streich2021`.

With the development of these advanced algorithms, however, the complexity of WFS software is steadily increasing as the field matures, which hinders cooperation as well as end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

What is OpenWFS?
----------------------

OpenWFS is a Python package for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:

    * **Spatial light modulator**. The :class:`~.slm.SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated by OpenGL.
    * **Scanning microscope**. The :class:`~.devices.ScanningMicroscope` object uses a National Instruments data acquisition card to control a laser-scanning microscope.
    * **GenICam cameras**. The :class:`~.devices.Camera` object uses the `harvesters` backend :cite:`harvesters` to access any camera supporting the GenICam standard :cite:`genicam`.
    * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e. g. an SLM) and detectors (e. g. a camera). The automatic synchronization makes it trivial to perform pipelined measurements that avoid the delay normally caused by the latency of the video card and SLM.

* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, most algorithms can be implemented cleanly without hardware-specific programming.

* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

* **Platform for exchange and joint collaboration**. OpenWFS can be used as a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools. 

* **Platform for simplifying use of wavefront shaping**. OpenWFS is compatible to the recently developed PyDevice :cite:`PyDevice`, and can therefore be controlled from Micro-Manager :cite:`MMoverview`, a commonly used microscopy control platform. 

* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

.. only:: latex

    Here, we first show how to get started using OpenWFS for simulating and controlling wavefront shaping experiments. An in-depth discussion of the core design of OpenWFS is given in :numref:`Key concepts`. Key to any wavefront shaping experiment is the SLM. The support for advanced options like texture warping and the use of a software lookup table are explained in :numref:`section-slms`.

    The ability to simulate optical experiments is essential for the rapid development and debugging of wavefront shaping algorithms. The built-in options for realistically simulating experiments are be discussed in :numref:`section-simulations`. Finally, OpenWFS is designed to be modular and easy to extend.  In :numref:`section-development`, we show how to write custom hardware control modules. Note that not all functionality of the package is covered in this document, and we refer to the API documentation :cite:`openwfsdocumentation` for a complete overview of most recent version of the package.


Getting started
----------------------
OpenWFS is available on the PyPI repository, and it can be installed with the command ``pip install openwfs``. The latest documentation and the example code can be found on the `Read the Docs <https://openwfs.readthedocs.io/en/latest/>`_ website :cite:`openwfsdocumentation`. To use OpenWFS, you need to have Python 3.9 or later installed. At the time of writing, OpenWFS is tested up to Python version 3.11 (not all dependencies were available for Python 3.12 yet). OpenWFS is developed and tested on Windows 11 and Manjaro Linux.

:numref:`hello-wfs` shows an example of how to use OpenWFS to run a simple wavefront shaping experiment. This example illustrates several of the main concepts of OpenWFS. First, the code initializes objects to control a spatial light modulator (SLM) connected to a video port, and a camera that provides feedback to the wavefront shaping algorithm.

.. _hello-wfs:
.. literalinclude:: ../../examples/hello_wfs.py
   :language: python
   :caption: ``hello_wfs.py``. Example of a simple wavefront shaping experiment using OpenWFS.

This example uses the `StepwiseSequential` wavefront shaping algorithm :cite:`vellekoop2008phase`. The algorithm needs access to the SLM for controlling the wavefront. This feedback is obtained from a  :class:`~.SingleRoi` object, which takes images from the camera, and averages them over the specified circular region of interest. The algorithm returns the measured transmission matrix in the field `results.t`, which is used to compute the optimal phase pattern to compensate the aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase pattern.

This code illustrates how OpenWFS separates the concerns of the hardware control (`SLM` and `Camera`), signal processing (`SingleROIProcessor`) and the algorithm itself (`StepwiseSequential`). A large variety of wavefront shaping experiments can be performed by using different types of feedback signals (such as optimizing multiple foci simultaneously using a :class:`~.MultiRoiProcessor` object), using different algorithms, or different image sources, such as a :class:`~.ScanningMicroscope`. Notably, these objects can be replaced by *mock* objects, that simulate the hardware and allow for rapid prototyping and testing of new algorithms without direct access to wavefront shaping hardware (see :numref:`section-simulations`).


Analysis and troubleshooting
------------------------------------------------
The principles of wavefront shaping are well established, and under close-to-ideal experimental conditions, it is possible to accurately predict the signal enhancement. In practice, however, there exist many practical issues that can negatively affect the outcome of the experiment. OpenWFS has built-in functions to analyze and troubleshoot the measurements from a wavefront shaping experiment.

The ``result`` structure in :numref:`hello-wfs`, as returned by the wavefront shaping algorithm, was computed with the utility function :func:`analyze_phase_stepping`. This function extracts the transmission matrix from phase stepping measurements, and additionally computes a series of troubleshooting statistics in the form of a *fidelity*, which is a number that ranges from 0 (no sensible measurement possible) to 1 (perfect situation, optimal focus expected). These fidelities are:

* :attr:`~.WFSResults.fidelity_noise`: The fidelity reduction due to noise in the measurements.
* :attr:`~.WFSResults.fidelity_amplitude`: The fidelity reduction due to unequal illumination of the SLM.
* :attr:`~.WFSResults.fidelity_calibration`: The fidelity reduction due to imperfect phase response of the SLM.

If these fidelities are much lower than 1, this indicates a problem in the experiment, or a bug in the wavefront shaping experiment. For a comprehensive overview of the practical considerations in wavefront shaping and their effects on the fidelity, please see :cite:`Mastiani2024PracticalConsiderations`.

Further troubleshooting can be performed with the :func:`~.troubleshoot` function, which estimates the following fidelities:

* :attr:`~.WFSTroubleshootResult.fidelity_non_modulated`: The fidelity reduction due to non-modulated light., e. g. due to reflection from the front surface of the SLM.
* :attr:`~.WFSTroubleshootResult.fidelity_decorrelation`: The fidelity reduction due to decorrelation of the field during the measurement.

All fidelity estimations are combined to make an order of magnitude estimation of the expected enhancement. :func:`~.troubleshoot` returns a ``WFSTroubleshootResult`` object containing the outcome of the different tests and analyses, which can be printed to the console as a comprehensive troubleshooting report with the method :meth:`~.WFSTroubleshootResult.report()`. See ``examples/troubleshooter_demo.py`` for an example of how to use the automatic troubleshooter.

Lastly, the :func:`~.troubleshoot` function computes several image frame metrics such as the *unbiased contrast to noise ratio* and *unbiased contrast enhancement*. These metrics are especially useful for scenarios where the contrast is expected to improve due to wavefront shaping, such as in multi-photon excitation fluorescence (multi-PEF) microscopy. Furthermore, :func:`~.troubleshoot` tests the image capturing repeatability and runs a stability test by capturing and comparing many frames over a longer period of time.

%endmatter%

