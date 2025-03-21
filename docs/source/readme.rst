.. only:: markdown
 
   .. raw:: html
   
      [![PyTest Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml)
      [![Black Code Style Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml)</p>

..
   NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY. 
    
..
      
What is wavefront shaping?
--------------------------------

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials :cite:`kubby2019`. In WFS, a spatial light modulator (SLM) is used to shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through :cite:`Vellekoop2007`, or inside :cite:`vellekoop2008demixing` scattering materials; or light can be shaped to have other desired properties, such as optimal sensitivity for specific measurements :cite:`bouchet2021maximum`, specialized point-spread functions :cite:`boniface2017transmission` or spectral filtering :cite:`Park2012`.

It stands out that an important driving force in WFS is the development of new algorithms, for example, to account for sample movement :cite:`valzania2023online`, experimental conditions :cite:`Anderson2016`, to be optimally resilient to noise :cite:`mastiani2021noise`, or to use digital twin models to compute the required correction patterns :cite:`salter2014exploring,ploschner2015seeing,Thendiyammal2020,cox2023model`. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed specifically for the methodology of wavefront shaping, such as using algorithms based on Hadamard patterns :cite:`popoff2010measuring` or Fourier-based approaches :cite:`Mastiani2022`. Fast techniques that enable wavefront shaping in dynamic samples :cite:`Liu2017,Tzang2019` have also been developed, and many potential applications have been prototyped, including endoscopy :cite:`ploschner2015seeing`, optical trapping :cite:`Cizmar2010`, Raman scattering :cite:`Thompson2016`, and deep-tissue imaging :cite:`Streich2021`. Applications extend beyond that of microscope imaging, such as in optimizing photoelectrochemical absorption :cite:`Liew2016` and tuning random lasers :cite:`Bachelard2014`.

With the development of these advanced algorithms, however, the complexity of WFS software is steadily increasing as the field matures, which hinders cooperation as well as end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

Even though authors are increasingly sharing their code, for example for controlling spatial light modulators (SLMs) :cite:`PopoffslmPy`, or running genetic algorithms :cite:`Anderson2024`, a modular framework that combines all aspects of hardware control, simulation, and graphical user interface (GUI) integration is still lacking.

What is OpenWFS?
----------------------

OpenWFS is a Python package that is primarily designed for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:

    * **Spatial light modulator**. The :class:`~.slm.SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture mapping, and texture blending functionality accelerated by OpenGL.
    * **Scanning microscope**. The :class:`~.devices.ScanningMicroscope` object uses a National Instruments data acquisition card to control a laser-scanning microscope.
    * **GenICam cameras**. The :class:`~.devices.Camera` object uses the ``harvesters`` backend :cite:`harvesters` to access any camera supporting the GenICam standard :cite:`genicam`.
    * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). The automatic synchronization makes it trivial to perform pipelined measurements :cite:`ThesisVellekoop` that avoid the delay normally caused by the latency of the video card and SLM.

* **Simulation**.  The ability to simulate optical experiments is essential for the rapid development and debugging of wavefront shaping algorithms. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms without the need for physical hardware.

* **Wavefront shaping algorithms**. A growing collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, even advanced algorithms can be implemented in a few dozen lines of code, and automatically work with any combination of hardware and simulation tools that OpenWFS supports.

* **Platform for exchange and joint collaboration**. OpenWFS can be used as a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools. Python was specifically chosen for this purpose for its active community, high level of abstraction and the ease of sharing tools.

* **Micro-Manager compatibility**. Micro-Manager :cite:`MMoverview` is a widely used open-source microscopy control platform. The devices in OpenWFS, such as GenICam camera's, or the scanning microscope, as well as all algorithms, can be controlled from Micro-Manager using the recently developed :cite:`PyDevice` adapter that imports Python scripts into Micro-Manager

* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.



.. only:: latex

    Here, we first show how to get started using OpenWFS for simulating and controlling wavefront shaping experiments. An in-depth discussion of the core design of OpenWFS is given in :numref:`section-key_concepts`. Key to any wavefront shaping experiment is the spatial light modulator. The support for advanced options like texture mapping and the use of a software lookup table are explained in :numref:`section-slms`. The tools for realistically simulating experiments, automatic troubleshooting of experiments, and Micro-Manager integration are discussed in :numref:`section-simulations`, :numref:`section-troubleshooting`, and :numref:`section-micromanager`, respectively. Finally, in :numref:`section-development`, we show how to write custom hardware control modules in order to extend the functionality of OpenWFS.

    Note that not all functionality of the package is covered in this document. We refer to the API documentation :cite:`openwfsdocumentation` for a complete overview of most recent version of the package.

Getting started
----------------------
To use OpenWFS, Python 3.9 or later is required. Since it is available on the PyPI repository, OpenWFS can be installed using ``pip``:

.. code-block:: bash

    pip install openwfs[all]

It is advised to make a new environment for OpenWFS, such as with conda, poetry, or Python's venv. This will also install the optional dependencies for OpenWFS:

*opengl* For the OpenGL-accelerated SLM control, the ``PyOpenGL`` package is installed. In order for this package to work, an OpenGL-compatible graphics card and driver is required.

*genicam* For the GenICam camera support, the ``harvesters`` package is installed, which, in turn, needs the  ``genicam`` package. At the time of writing, this package is only available for Python versions up to 3.11. To use the GenICam camera support, you also need a compatible camera with driver installed.

*nidaq* For the scanning microscope, the ``nidaqmx`` package is installed, which requires a National Instruments data acquisition card with corresponding drivers to be installed on your system.

If these dependencies cannot be installed on your system, the installation will fail. In this case, you can instead install OpenWFS without dependencies by omitting ``[all]`` in the installation command, and manually install only the required dependencies, e.g. ``pip install openwfs[opengl,nidaq]``.

At the time of writing, OpenWFS is at version 1.0.0, and it is tested up to Python version 3.11 on Windows 11 and Manjaro and Ubuntu Linux distributions. Note that the latest versions of the package will be available on the PyPI repository, and the latest documentation and the example code can be found on the `Read the Docs <https://openwfs.readthedocs.io/en/latest/>`_ website :cite:`openwfsdocumentation`. The source code can be found on :cite:`openwfsgithub`.

:numref:`hello-wfs` shows an example of how to use OpenWFS to run a simple wavefront shaping experiment. This example illustrates several of the main concepts of OpenWFS. First, the code initializes objects to control a spatial light modulator (SLM) connected to a video port, and a camera that provides feedback to the wavefront shaping algorithm. It then runs a WFS algorithm to focus the light.

This example uses the :class:`~.StepwiseSequential` wavefront shaping algorithm :cite:`vellekoop2008phase`. The algorithm needs access to the SLM for controlling the wavefront. This feedback is obtained from a  :class:`~.SingleRoi` object, which takes images from the camera, and averages them over the specified circular region of interest. The algorithm returns the measured transmission matrix in the field `results.t`, which is used to compute the optimal phase pattern to compensate the aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase pattern.

.. _hello-wfs:
.. literalinclude:: ../../examples/hello_wfs.py
   :language: python
   :caption: ``hello_wfs.py``. Example of a simple wavefront shaping experiment using OpenWFS.

This code illustrates how OpenWFS separates the concerns of the hardware control (:class:`~.SLM` and :class:`~.Camera`), signal processing (:class:`~.SingleRoi`) and the algorithm itself (:class:`~.StepwiseSequential`). A large variety of wavefront shaping experiments can be performed by using different types of feedback signals (such as optimizing multiple foci simultaneously using a :class:`~.MultiRoi` object), using different algorithms, or different image sources, such as a :class:`~.ScanningMicroscope`. Notably, these objects can be replaced by *mock* objects, that simulate the hardware and allow for rapid prototyping and testing of new algorithms without direct access to wavefront shaping hardware (see :numref:`section-simulations`).


%endmatter%

