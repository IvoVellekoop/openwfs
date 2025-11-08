[![PyTest Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml)
[![Black Code Style Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml)</p>
<!-- NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY. Instead, edit readme.rst and generate README.md by running the command `./make.bat markdown` from the `docs/source` folder. -->

# What is wavefront shaping?

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials [[1](#id84)]. In WFS, a spatial light modulator (SLM) is used to shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through [[2](#id62)], or inside [[3](#id51)] scattering materials; or light can be shaped to have other desired properties, such as optimal sensitivity for specific measurements [[4](#id52)], specialized point-spread functions [[5](#id39)] or spectral filtering [[6](#id77)].

It stands out that an important driving force in WFS is the development of new algorithms, for example, to account for sample movement [[7](#id41)], experimental conditions [[8](#id70)], to be optimally resilient to noise [[9](#id40)], or to use digital twin models to compute the required correction patterns [[10](#id61), [11](#id60), [12](#id79), [13](#id43)]. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed specifically for the methodology of wavefront shaping, such as using algorithms based on Hadamard patterns [[14](#id35)] or Fourier-based approaches [[15](#id56)]. Fast techniques that enable wavefront shaping in dynamic samples [[16](#id65), [17](#id66)] have also been developed, and many potential applications have been prototyped, including endoscopy [[11](#id60)], optical trapping [[18](#id67)], Raman scattering [[19](#id50)], and deep-tissue imaging [[20](#id68)]. Applications extend beyond that of microscope imaging, such as in optimizing photoelectrochemical absorption [[21](#id69)] and tuning random lasers [[22](#id76)].

With the development of these advanced algorithms, however, the complexity of WFS software is steadily increasing as the field matures, which hinders cooperation as well as end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

Even though authors are increasingly sharing their code, for example for controlling spatial light modulators (SLMs) [[23](#id64)], or running genetic algorithms [[24](#id75)], a modular framework that combines all aspects of hardware control, simulation, and graphical user interface (GUI) integration is still lacking.

# What is OpenWFS?

OpenWFS is a Python package that is primarily designed for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:
  > * **Spatial light modulator**. The `SLM` object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture mapping, and texture blending functionality accelerated by OpenGL.
  > * **Scanning microscope**. The `ScanningMicroscope` object uses a National Instruments data acquisition card to control a laser-scanning microscope.
  > * **GenICam cameras**. The `Camera` object uses the `harvesters` backend [[25](#id44)] to access any camera supporting the GenICam standard [[26](#id47)].
  > * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). The automatic synchronization makes it trivial to perform pipelined measurements [[27](#id37)] that avoid the delay normally caused by the latency of the video card and SLM.
* **Simulation**.  The ability to simulate optical experiments is essential for the rapid development and debugging of wavefront shaping algorithms. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms without the need for physical hardware.
* **Wavefront shaping algorithms**. A growing collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, even advanced algorithms can be implemented in a few dozen lines of code, and automatically work with any combination of hardware and simulation tools that OpenWFS supports.
* **Platform for exchange and joint collaboration**. OpenWFS can be used as a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools. Python was specifically chosen for this purpose for its active community, high level of abstraction and the ease of sharing tools.
* **Micro-Manager compatibility**. Micro-Manager [[28](#id73)] is a widely used open-source microscopy control platform. The devices in OpenWFS, such as GenICam camera’s, or the scanning microscope, as well as all algorithms, can be controlled from Micro-Manager using the recently developed [[29](#id85)] adapter that imports Python scripts into Micro-Manager
* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

# Getting started

To use OpenWFS, Python 3.9 or later is required. Since it is available on the PyPI repository, OpenWFS can be installed using `pip`:

```bash
pip install openwfs[all]
```

It is advised to make a new environment for OpenWFS, such as with conda, poetry, or Python’s venv. This will also install the optional dependencies for OpenWFS:

*opengl* For the OpenGL-accelerated SLM control, the `PyOpenGL` package is installed. In order for this package to work, an OpenGL-compatible graphics card and driver is required.

*genicam* For the GenICam camera support, the `harvesters` package is installed, which, in turn, needs the  `genicam` package. At the time of writing, this package is only available for Python versions up to 3.11. To use the GenICam camera support, you also need a compatible camera with driver installed.

*nidaq* For the scanning microscope, the `nidaqmx` package is installed, which requires a National Instruments data acquisition card with corresponding drivers to be installed on your system.

If these dependencies cannot be installed on your system, the installation will fail. In this case, you can instead install OpenWFS without dependencies by omitting `[all]` in the installation command, and manually install only the required dependencies, e.g. `pip install openwfs[opengl,nidaq]`.

At the time of writing, OpenWFS is at version 1.0.0, and it is tested up to Python version 3.11 on Windows 11 and Manjaro and Ubuntu Linux distributions. Note that the latest versions of the package will be available on the PyPI repository, and the latest documentation and the example code can be found on the [Read the Docs](https://openwfs.readthedocs.io/en/latest/) website [[30](#id74)]. The source code can be found on [[31](#id46)].

[Listing 3.1](#hello-wfs) shows an example of how to use OpenWFS to run a simple wavefront shaping experiment. This example illustrates several of the main concepts of OpenWFS. First, the code initializes objects to control a spatial light modulator (SLM) connected to a video port, and a camera that provides feedback to the wavefront shaping algorithm. It then runs a WFS algorithm to focus the light.

This example uses the `StepwiseSequential` wavefront shaping algorithm [[32](#id63)]. The algorithm needs access to the SLM for controlling the wavefront. This feedback is obtained from a  `SingleRoi` object, which takes images from the camera, and averages them over the specified circular region of interest. The algorithm returns the measured transmission matrix in the field results.t, which is used to compute the optimal phase pattern to compensate the aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase pattern.

<a id="hello-wfs"></a>
```python
"""
Hello wavefront shaping
===============================================
This script demonstrates how to use OpenWFS to perform a simple
wavefront shaping experiment. To run this script, you need to have
a GenICam-compatible camera connected to your computer,
and a spatial light modulator (SLM) connected to the secondary
video output.
"""

import numpy as np

from openwfs.algorithms import StepwiseSequential
from openwfs.devices import SLM, Camera
from openwfs.processors import SingleRoi

# Display the SLM patterns on the secondary monitor
slm = SLM(monitor_id=2)

# Connect to a GenICam camera, average pixels to get feedback signal
camera = Camera(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
feedback = SingleRoi(camera, pos=(320, 320), mask_type="disk", radius=2.5)

# Run the algorithm
alg = StepwiseSequential(feedback=feedback, slm=slm, n_x=10, n_y=10, phase_steps=4)
result = alg.execute()

# Measure intensity with flat and shaped wavefronts
slm.set_phases(0)
before = feedback.read()
slm.set_phases(-np.angle(result.t))
after = feedback.read()
print(f"Intensity in the target increased from  {before} to {after}")
```

This code illustrates how OpenWFS separates the concerns of the hardware control (`SLM` and `Camera`), signal processing (`SingleRoi`) and the algorithm itself (`StepwiseSequential`). A large variety of wavefront shaping experiments can be performed by using different types of feedback signals (such as optimizing multiple foci simultaneously using a `MultiRoi` object), using different algorithms, or different image sources, such as a `ScanningMicroscope`. Notably, these objects can be replaced by *mock* objects, that simulate the hardware and allow for rapid prototyping and testing of new algorithms without direct access to wavefront shaping hardware (see `section-simulations`).

# Developing OpenWFS

Read docsdevelopment.rst for more information on how to install a development version of OpenWFS, and how to write your own extensions.

# Acknowledgements

We would like to thank Gerwin Osnabrugge, Bahareh Mastiani, Giulia Sereni, Siebe Meijer, Gijs Hannink, Merle van Gorsel, Michele Gintoli, Karina van Beek, Abhilash Thendiyammal, Lyuba Amitonova, and Tzu-Lun Wang for their contributions to earlier revisions of our wavefront shaping code. This work was supported by the European Research Council under the European Union’s Horizon 2020 Programme / ERC Grant Agreement n° [678919], and the Dutch Research Council (NWO) through Vidi grant number 14879.
