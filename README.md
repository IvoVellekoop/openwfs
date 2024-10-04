[![PyTest Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/pytest.yml)
[![Black Code Style Status](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml/badge.svg)](https://github.com/IvoVellekoop/openwfs/actions/workflows/black.yml)</p>
<!-- NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY. -->

# What is wavefront shaping?

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures,
including strongly scattering materials [[1](#id76)]. In WFS, a spatial light modulator (SLM) is used to shape the phase
and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus
through [[2](#id57)], or inside [[3](#id46)] scattering materials; or light can be shaped to have other desired
properties, such as optimal sensitivity for specific measurements [[4](#id47)], specialized point-spread
functions [[5](#id33)], spectral filtering [[6](#id69)],, or for functions like optical trapping [[7](#id36)].

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for
sample movement [[8](#id35)], experimental conditions [[9](#id64)], to be optimally resilient to noise [[10](#id34)], or
to use digital twin models to compute the required correction
patterns [[11](#id56), [12](#id55), [13](#id72), [14](#id38)]. Much progress has been made towards developing fast and
noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as
using algorithms based on Hadamard patterns, or Fourier-based approaches [[15](#id51)]. Fast techniques that enable
wavefront shaping in dynamic samples [[16](#id59), [17](#id60)], and many potential applications have been developed and
prototyped, including endoscopy [[12](#id55)], optical trapping [[18](#id61)], Raman scattering, [[19](#id45)], and
deep-tissue imaging [[20](#id62)]. Applications extend beyond that of microscope imaging such as optimizing
photoelectrochemical absorption [[21](#id63)] and tuning random lasers [[22](#id68)].

With the development of these advanced algorithms, however, the complexity of WFS software is steadily increasing as the
field matures, which hinders cooperation as well as end-user adoption. Code for controlling wavefront shaping tends to
be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level
programming. A recent c++ based contribution [[23](#id67)], highlights the growing need for software based tools that
enable use and development. Moreover, since many labs use their own in-house programs to control the experiments,
sharing and re-using code between different research groups is troublesome.

# What is OpenWFS?

OpenWFS is a Python package for performing and for simulating wavefront shaping experiments. It aims to accelerate
wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically
  encountered in wavefront shaping experiments. Highlights include:
  > * **Spatial light modulator**. The `SLM` object provides a versatile way to control spatial light modulators,
      allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated
      by OpenGL.
  > * **Scanning microscope**. The `ScanningMicroscope` object uses a National Instruments data acquisition card to
      control a laser-scanning microscope.
  > * **GenICam cameras**. The `Camera` object uses the harvesters backend [[24](#id39)] to access any camera supporting
      the GenICam standard [[25](#id42)].
  > * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and
      detectors (e.g. a camera). The automatic synchronization makes it trivial to perform pipelined measurements that
      avoid the delay normally caused by the latency of the video card and SLM.
* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the
  hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a
  result, most algorithms can be implemented cleanly without hardware-specific programming.
* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms,
  including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid
  prototyping and testing of new algorithms, without the need for physical hardware.
* **Platform for exchange and joint collaboration**. OpenWFS can be used as a platform for sharing and exchanging
  wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the
  community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools.
  Python was specifically chosen for this purpose for its active community, high level of abstraction and the ease of
  sharing tools. Further expansion of the supported hardware is of high priority, especially wrapping c-based software
  support with tools like ctypes and the Micro-Manager based device adapters.
* **Platform for simplifying use of wavefront shaping**. OpenWFS is compatible with the recently developed PyDevice [],
  and can therefore be controlled from Micro-Manager [[26](#id65)], a commonly used microscopy control platform.
* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments.
  This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems
  such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.

# Getting started

OpenWFS is available on the PyPI repository, and it can be installed with the command `pip install openwfs`. The latest
documentation and the example code can be found on the [Read the Docs](https://openwfs.readthedocs.io/en/latest/)
website [[27](#id66)]. To use OpenWFS, you need to have Python 3.9 or later installed. At the time of writing, OpenWFS
is tested up to Python version 3.11 (not all dependencies were available for Python 3.12 yet). OpenWFS is developed and
tested on Windows 11 and Manjaro Linux.

[Listing 3.1](#hello-wfs) shows an example of how to use OpenWFS to run a simple wavefront shaping experiment. This
example illustrates several of the main concepts of OpenWFS. First, the code initializes objects to control a spatial
light modulator (SLM) connected to a video port, and a camera that provides feedback to the wavefront shaping algorithm.

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

This example uses the StepwiseSequential wavefront shaping algorithm [[28](#id58)]. The algorithm needs access to the
SLM for controlling the wavefront. This feedback is obtained from a  `SingleRoi` object, which takes images from the
camera, and averages them over the specified circular region of interest. The algorithm returns the measured
transmission matrix in the field results.t, which is used to compute the optimal phase pattern to compensate the
aberrations. Finally, the code measures the intensity at the detector before and after applying the optimized phase
pattern.

This code illustrates how OpenWFS separates the concerns of the hardware control (SLM and Camera), signal processing (
SingleROIProcessor) and the algorithm itself (StepwiseSequential). A large variety of wavefront shaping experiments can
be performed by using different types of feedback signals (such as optimizing multiple foci simultaneously using
a `MultiRoiProcessor` object), using different algorithms, or different image sources, such as a `ScanningMicroscope`.
Notably, these objects can be replaced by *mock* objects, that simulate the hardware and allow for rapid prototyping and
testing of new algorithms without direct access to wavefront shaping hardware (see `section-simulations`).

# Analysis and troubleshooting

The principles of wavefront shaping are well established, and under close-to-ideal experimental conditions, it is
possible to accurately predict the signal enhancement. In practice, however, there exist many practical issues that can
negatively affect the outcome of the experiment. OpenWFS has built-in functions to analyze and troubleshoot the
measurements from a wavefront shaping experiment.

The `result` structure in [Listing 3.1](#hello-wfs), as returned by the wavefront shaping algorithm, was computed with
the utility function `analyze_phase_stepping()`. This function extracts the transmission matrix from phase stepping
measurements, and additionally computes a series of troubleshooting statistics in the form of a *fidelity*, which is a
number that ranges from 0 (no sensible measurement possible) to 1 (perfect situation, optimal focus expected). These
fidelities are:

* `fidelity_noise`: The fidelity reduction due to noise in the measurements.
* `fidelity_amplitude`: The fidelity reduction due to unequal illumination of the SLM.
* `fidelity_calibration`: The fidelity reduction due to imperfect phase response of the SLM.

If these fidelities are much lower than 1, this indicates a problem in the experiment, or a bug in the wavefront shaping
experiment. For a comprehensive overview of the practical considerations in wavefront shaping and their effects on the
fidelity, please see [[29](#id31)].

Further troubleshooting can be performed with the `troubleshoot()` function, which estimates the following fidelities:

* `fidelity_non_modulated`: The fidelity reduction due to non-modulated light., e.g. due to reflection from the front
  surface of the SLM.
* `fidelity_decorrelation`: The fidelity reduction due to decorrelation of the field during the measurement.

All fidelity estimations are combined to make an order of magnitude estimation of the expected
enhancement. `troubleshoot()` returns a `WFSTroubleshootResult` object containing the outcome of the different tests and
analyses, which can be printed to the console as a comprehensive troubleshooting report with the method `report()`.
See `examples/troubleshooter_demo.py` for an example of how to use the automatic troubleshooter.

Lastly, the `troubleshoot()` function computes several image frame metrics such as the *unbiased contrast to noise
ratio* and *unbiased contrast enhancement*. These metrics are especially useful for scenarios where the contrast is
expected to improve due to wavefront shaping, such as in multi-photon excitation fluorescence (multi-PEF) microscopy.
Furthermore, `troubleshoot()` tests the image capturing repeatability and runs a stability test by capturing and
comparing many frames over a longer period of time.

# Acknowledgements

We would like to thank Gerwin Osnabrugge, Bahareh Mastiani, Giulia Sereni, Siebe Meijer, Gijs Hannink, Merle van Gorsel,
Michele Gintoli, Karina van Beek, Abhilash Thendiyammal, Lyuba Amitonova, and Tzu-Lun Wang for their contributions to
earlier revisions of our wavefront shaping code. This work was supported by the European Research Council under the
European Union’s Horizon 2020 Programme / ERC Grant Agreement n° [678919], and the Dutch Research Council (NWO) through
Vidi grant number 14879.

# Conflict of interest statement

The authors declare no conflict of interest.

<a name="id76"></a>1

Joel Kubby, Sylvain Gigan, and Meng Cui, editors. *Wavefront Shaping for Biomedical Imaging*. Advances in Microscopy and
Microanalysis. Cambridge University Press, 2019. [doi:10.1017/9781316403938](https://doi.org/10.1017/9781316403938).

<a name="id57"></a>2

Ivo M. Vellekoop and A. P. Mosk. Focusing coherent light through opaque strongly scattering media. *Opt. Lett.*, 32(16):
2309–2311, Aug 2007. [doi:10.1364/OL.32.002309](https://doi.org/10.1364/OL.32.002309).

<a name="id46"></a>3

Ivo M. Vellekoop, EG Van Putten, A Lagendijk, and AP Mosk. Demixing light paths inside disordered metamaterials. *Optics
express*, 16(1):67–80, 2008.

<a name="id47"></a>4

Dorian Bouchet, Stefan Rotter, and Allard P Mosk. Maximum information states for coherent scattering measurements.
*Nature Physics*, 17(5):564–568, 2021.

<a name="id33"></a>5

Antoine Boniface et al. Transmission-matrix-based point-spread-function engineering through a complex medium. *Optica*,
4(1):54–59, 2017.

<a name="id69"></a>6

Jung-Hoon Park, ChungHyun Park, YongKeun Park, Hyunseung Yu, and Yong-Hoon Cho. Active spectral filtering through turbid
media. *Optics Letters, Vol. 37, Issue 15, pp. 3261-3263*, 37:3261–3263, 8 2012.
URL: [https://opg.optica.org/viewmedia.cfm?uri=ol-37-15-3261&seq=0&html=true https://opg.optica.org/abstract.cfm?uri=ol-37-15-3261 https://opg.optica.org/ol/abstract.cfm?uri=ol-37-15-3261](https://opg.optica.org/viewmedia.cfm?uri=ol-37-15-3261&seq=0&html=true https://opg.optica.org/abstract.cfm?uri=ol-37-15-3261 https://opg.optica.org/ol/abstract.cfm?uri=ol-37-15-3261), [doi:10.1364/OL.37.003261](https://doi.org/10.1364/OL.37.003261).

<a name="id36"></a>7

Tomáš Čižmár, Michael Mazilu, and Kishan Dholakia. In situ wavefront correction and its application to
micromanipulation. *Nature Photonics*, 4(6):388–394, 2010.

<a name="id35"></a>8

Lorenzo Valzania and Sylvain Gigan. Online learning of the transmission matrix of dynamic scattering media. *Optica*,
10(6):708–716, 2023.

<a name="id64"></a>9

Benjamin R. Anderson, Ray Gunawidjaja, and Hergen Eilers. Effect of experimental parameters on optimal reflection of
light from opaque media. *Physical Review A*, 93:013813, 1 2016.
URL: [https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.013813](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.013813), [doi:10.1103/PHYSREVA.93.013813/FIGURES/12/MEDIUM](https://doi.org/10.1103/PHYSREVA.93.013813/FIGURES/12/MEDIUM).

<a name="id34"></a>10

Bahareh Mastiani and Ivo M Vellekoop. Noise-tolerant wavefront shaping in a hadamard basis. *Optics express*, 29(11):
17534–17541, 2021.

<a name="id56"></a>11

PS Salter, M Baum, I Alexeev, M Schmidt, and MJ Booth. Exploring the depth range for three-dimensional laser machining
with aberration correction. *Optics express*, 22(15):17644–17656, 2014.

<a name="id55"></a>12

Martin Plöschner, Tomáš Tyc, and Tomáš Čižmár. Seeing through chaos in multimode fibres. *Nature Photonics*, 9(8):
529–535, 2015.

<a name="id72"></a>13

Abhilash Thendiyammal, Gerwin Osnabrugge, Tom Knop, and Ivo M. Vellekoop. Model-based wavefront shaping microscopy.
*Opt. Lett.*, 45(18):5101–5104, Sep 2020. [doi:10.1364/OL.400985](https://doi.org/10.1364/OL.400985).

<a name="id38"></a>14

DWS Cox, T Knop, and Ivo M. Vellekoop. Model-based aberration corrected microscopy inside a glass tube. *arXiv preprint
arXiv:2311.13363*, 2023.

<a name="id51"></a>15

Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop. Wavefront shaping for forward scattering. *Optics Express*,
30:37436, 10 2022. [doi:10.1364/oe.470194](https://doi.org/10.1364/oe.470194).

<a name="id59"></a>16

Yan Liu et al. Focusing light inside dynamic scattering media with millisecond digital optical phase conjugation.
*Optica*, 4(2):280–288, Feb 2017. [doi:10.1364/OPTICA.4.000280](https://doi.org/10.1364/OPTICA.4.000280).

<a name="id60"></a>17

Omer Tzang et al. Wavefront shaping in complex media with a 350 khz modulator via a 1d-to-2d transform. *Nature
Photonics*, 2019. [doi:10.1038/s41566-019-0503-6](https://doi.org/10.1038/s41566-019-0503-6).

<a name="id61"></a>18

Tomáš Čižmár, Michael Mazilu, and Kishan Dholakia. In situ wavefront correction and its application to
micromanipulation. *Nature Photonics*, 4:388–394, 05
2010. [doi:10.1038/nphoton.2010.85](https://doi.org/10.1038/nphoton.2010.85).

<a name="id45"></a>19

Jonathan V. Thompson, Graham A. Throckmorton, Brett H. Hokr, and Vladislav V. Yakovlev. Wavefront shaping enhanced raman
scattering in a turbid medium. *Optics letters*, 41:1769, 4 2016.
URL: [https://pubmed.ncbi.nlm.nih.gov/27082341/](https://pubmed.ncbi.nlm.nih.gov/27082341/), [doi:10.1364/OL.41.001769](https://doi.org/10.1364/OL.41.001769).

<a name="id62"></a>20

Lina Streich et al. High-resolution structural and functional deep brain imaging using adaptive optics three-photon
microscopy. *Nature Methods 2021 18:10*, 18:1253–1258, 9
2021. [doi:10.1038/s41592-021-01257-6](https://doi.org/10.1038/s41592-021-01257-6).

<a name="id63"></a>21

Seng Fatt Liew, Sébastien M. Popoff, Stafford W. Sheehan, Arthur Goetschy, Charles A. Schmuttenmaer, A. Douglas Stone,
and Hui Cao. Coherent control of photocurrent in a strongly scattering photoelectrochemical system. *ACS Photonics*, 3:
449–455, 3 2016.
URL: [https://technion-staging.elsevierpure.com/en/publications/coherent-control-of-photocurrent-in-a-strongly-scattering-photoel](https://technion-staging.elsevierpure.com/en/publications/coherent-control-of-photocurrent-in-a-strongly-scattering-photoel), [doi:10.1021/ACSPHOTONICS.5B00642](https://doi.org/10.1021/ACSPHOTONICS.5B00642).

<a name="id68"></a>22

Nicolas Bachelard, Sylvain Gigan, Xavier Noblin, and Patrick Sebbah. Adaptive pumping for spectral control of random
lasers. *Nature Physics*, 10:426–431, 2014.
URL: [https://ui.adsabs.harvard.edu/abs/2014NatPh..10..426B/abstract](https://ui.adsabs.harvard.edu/abs/2014NatPh..10..426B/abstract), [doi:10.1038/nphys2939](https://doi.org/10.1038/nphys2939).

<a name="id67"></a>23

Benjamin R. Anderson, Andrew O’Kins, Kostiantyn Makrasnov, Rebecca Udby, Patrick Price, and Hergen Eilers. A modular
gui-based program for genetic algorithm-based feedback-assisted wavefront shaping. *Journal of Physics: Photonics*, 6:
045008, 8 2024.
URL: [https://iopscience.iop.org/article/10.1088/2515-7647/ad6ed3 https://iopscience.iop.org/article/10.1088/2515-7647/ad6ed3/meta](https://iopscience.iop.org/article/10.1088/2515-7647/ad6ed3 https://iopscience.iop.org/article/10.1088/2515-7647/ad6ed3/meta), [doi:10.1088/2515-7647/AD6ED3](https://doi.org/10.1088/2515-7647/AD6ED3).

<a name="id39"></a>24

Rod Barman et al. Harvesters. URL: [https://github.com/genicam/harvesters](https://github.com/genicam/harvesters).

<a name="id42"></a>25

GenICam - generic interface for cameras.
URL: [https://www.emva.org/standards-technology/genicam/](https://www.emva.org/standards-technology/genicam/).

<a name="id65"></a>26

Mark Tsuchida and Sam Griffin. Micro-manager project overview.
URL: [https://micro-manager.org/Micro-Manager_Project_Overview](https://micro-manager.org/Micro-Manager_Project_Overview).

<a name="id66"></a>27

OpenWFS documentation. URL: [https://openwfs.readthedocs.io/en/latest/](https://openwfs.readthedocs.io/en/latest/).

<a name="id58"></a>28

Ivo M. Vellekoop and AP Mosk. Phase control algorithms for focusing light through turbid media. *Optics communications*,
281(11):3071–3080, 2008.

<a name="id31"></a>29

Bahareh Mastiani, Daniël W. S. Cox, and Ivo M. Vellekoop. Practical considerations for high-fidelity wavefront shaping
experiments. http://arxiv.org/abs/2403.15265, March
2024. [arXiv:2403.15265](https://arxiv.org/abs/2403.15265), [doi:10.48550/arXiv.2403.15265](https://doi.org/10.48550/arXiv.2403.15265).
