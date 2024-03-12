OpenWFS - a library for conducting and simulating wavefront shaping experiments
=====================================================================================================================================================


What is wavefront shaping?
--------------------------------

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials. In WFS, a spatial light modulator (SLM) is used to spatially shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through :cite:`Vellekoop2007`, or inside :cite:`vellekoop2008demixing` a scattering material
in such a way that the light interferes constructively at the desired focus; or light can be shaped to have other desired properties, such an optimal sensitivity for measurements :cite:`bouchet2021maximum`, specialized point-spread functions :cite:`boniface2017transmission` or for functions like optical trapping :cite:`vcivzmar2010situ`.

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for sample movement :cite:`valzania2023online`, to be optimally resilient to noise :cite:`mastiani2021noise`, or to use digital twin models to compute the required correction patterns :cite:`salter2014exploring,ploschner2015seeing,Thendiyammal2020,cox2023model`. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as using algorithms based on Hadamard pattern, or Fourier-based approaches :cite:`Mastiani2022`, Fast techniques that enable wavefront shaping in dynamic samples :cite:`Liu2017` :cite:`Tzang2019`, and many potential applications have been developed and prototyped, including endoscopy, optical trapping :cite:`Cizmar2010` and deep-tissue imaging :cite:`Streich2021`.

With the development of these advanced algorithms, however, the  complexity of WFS software is gradually is becoming a bottleneck for further developments in the field, as well as for end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

What is OpenWFS?
----------------------

OpenWFS aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:

    * **Spatial light modulator**. The SLM :class:`~.slm.SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated by OpenGL.
    * **Scanning microscope**. The :class:`~.devices.ScanningMicroscope` object uses National Instruments Data Acquisition Cards to control a laser-scanning microscope.
    * **GeniCam cameras**. The :class:`~.devices.Camera` object uses the `harvesters` backend to access any camera supporting the GeniCam standard.
    * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). See :ref:`synchronization`. The automatic synchronization makes it trivial to perform pipelined measurements that avoid the delay normally caused by the latency of the video card and SLM.

* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, most algorithms can be implemented in just a few lines of code without the need for low-level or hardware-specific programming.

* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.

* **Platform for exchange and joint collaboration**. OpenWFS is designed to be a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools.

* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

* **MicroManager integration** (work in progress).  This code is designed so that it can be used in conjunction with `Micro-manager <https://micro-manager.org/>`_, a free and open-source microscopy, without any modification. To use this code in MicroManager, you need the PyDevice plugin, which can be found here:
    https://www.github.com/IvoVellekoop/pydevice


Getting started
--------------------
OpenWFS is available on PyPI, and can be installed using pip:

``pip install openwfs``

Note: due to compatibility issues with the genicam package, OpenWFS currently only works with Python 3.9-3.11.

To run the examples, create a local directory and clone the repository from GitHub using

``git clone http://www.github.com/IvoVellekoop/openwfs.git``

The examples are located in the openwfs/examples folder.
Dependencies can be automatically installed with `Poetry <https://python-poetry.org/>`_ by running

``poetry install --with dev --with docs``

from the openwfs directory. This also installs the optional dependencies for running the tests
and for building the documentation using Sphinx. The tests (in `/tests`) and examples in (`/examples`)
provide a good starting point for learning how to use OpenWFS.


Bibliography
--------------------
.. bibliography::
