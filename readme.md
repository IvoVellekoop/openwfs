# OpenWFS - a library for conducting and simulating wavefront shaping experiments

<!-- NOTE: README.MD IS AUTO-GENERATED FROM README.RST. DO NOT EDIT README.MD DIRECTLY. -->

## What is wavefront shaping?

Wavefront shaping (WFS) is a technique for controlling the propagation of light in arbitrarily complex structures, including strongly scattering materials. In WFS, a spatial light modulator (SLM) is used to spatially shape the phase and/or amplitude of the incident light. With a properly constructed wavefront, light can be made to focus through [[1](#id30)], or inside [[2](#id21)] a scattering material in such a way that the light interferes constructively at the desired focus; or light can be shaped to have other desired properties, such an optimal sensitivity for measurements [[3](#id22)], specialized point-spread functions [[4](#id16)] or for functions like optical trapping [[5](#id19)].

It stands out that an important driving force in WFS is the development of new algorithms, for example to account for sample movement [[6](#id18)], to be optimally resilient to noise [[7](#id17)], or to use digital twin models to compute the required correction patterns [[8](#id29), [9](#id28), [10](#id37), [11](#id20)]. Much progress has been made towards developing fast and noise-resilient algorithms, or algorithms designed for specific towards the methodology of wavefront shaping, such as using algorithms based on Hadamard pattern, or Fourier-based approaches [[12](#id24)], Fast techniques that enable wavefront shaping in dynamic samples [[13](#id31)] [[14](#id32)], and many potential applications have been developed and prototyped, including endoscopy, optical trapping [[15](#id33)] and deep-tissue imaging [[16](#id34)].

With the development of these advanced algorithms, however, the  complexity of WFS software is gradually is becoming a bottleneck for further advancements in the field, as well as for end-user adoption. Code for controlling wavefront shaping tends to be complex and setup-specific, and developing this code typically requires detailed technical knowledge and low-level programming. Moreover, since many labs use their own in-house programs to control the experiments, sharing and re-using code between different research groups is troublesome.

## What is OpenWFS?

OpenWFS is a Python package for performing and for simulating wavefront shaping experiments. It aims to accelerate wavefront shaping research by providing:

* **Hardware control**. Modular code for controlling spatial light modulators, cameras, and other hardware typically encountered in wavefront shaping experiments. Highlights include:
  > * **Spatial light modulator**. The SLM [`SLM`](slm.md#openwfs.slm.SLM) object provides a versatile way to control spatial light modulators, allowing for software lookup tables, synchronization, texture warping, and multi-texture functionality accelerated by OpenGL.
  > * **Scanning microscope**. The [`ScanningMicroscope`](devices.md#openwfs.devices.ScanningMicroscope) object uses National Instruments Data Acquisition Cards to control a laser-scanning microscope.
  > * **GeniCam cameras**. The [`Camera`](devices.md#openwfs.devices.Camera) object uses the harvesters backend to access any camera supporting the GeniCam standard.
  > * **Automatic synchronization**. OpenWFS provides tools for automatic synchronization of actuators (e.g. an SLM) and detectors (e.g. a camera). See [Synchronization](detectors_and_actuators.md#synchronization). The automatic synchronization makes it trivial to perform pipelined measurements that avoid the delay normally caused by the latency of the video card and SLM.
* **Wavefront shaping algorithms**. A (growing) collection of wavefront shaping algorithms. OpenWFS abstracts the hardware control, synchronization, and signal processing so that the user can focus on the algorithm itself. As a result, most algorithms can be implemented in just a few lines of code without the need for low-level or hardware-specific programming.
* **Simulation**. OpenWFS provides an extensive framework for testing and simulating wavefront shaping algorithms, including the effect of measurement noise, stage drift, and user-defined aberrations. This allows for rapid prototyping and testing of new algorithms, without the need for physical hardware.
* **Platform for exchange and joint collaboration**. OpenWFS is designed to be a platform for sharing and exchanging wavefront shaping algorithms. The package is designed to be modular and easy to expand, and it is our hope that the community will contribute to the package by adding new algorithms, hardware control modules, and simulation tools.
* **Automated troubleshooting**. OpenWFS provides tools for automated troubleshooting of wavefront shaping experiments. This includes tools for measuring the performance of wavefront shaping algorithms, and for identifying common problems such as incorrect SLM calibration, drift, measurement noise, and other experimental imperfections.
* **MicroManager integration** (work in progress).  This code is designed so that it can be used in conjunction with [Micro-manager](https://micro-manager.org/), a free and open-source microscopy, without any modification. To use this code in MicroManager, you need the PyDevice plugin, which can be found here:
  : [https://www.github.com/IvoVellekoop/pydevice](https://www.github.com/IvoVellekoop/pydevice)

OpenWFS is available on the PyPi repository, and the latest documentation can be found on [Read the Docs](https://openwfs.readthedocs.io/en/latest/).

In this documentation, we first show how to get started with OpenWFS by  can be used to simulate and control simple wavefront shaping experiments (see [Getting started](#getting-started)).

# Getting started

OpenWFS is available on PyPI, and can be installed using pip[1]:
[1]: due to compatibility issues with the ``genicam`` package, OpenWFS currently only works with Python versions 3.9-3.11.

`pip install openwfs`

The documentation for the latest version can be found on [online](https://openwfs.readthedocs.io/en/latest/).
After installing this package, all examples in the documentation can be run. For components that control actual hardware,
however, it may be needed to install additional drivers, as detailed in the documentation for these components.

## Installing for development, running tests and examples

To install the full source code, including examples, unit tests, and documentation source files, create a local directory and clone the repository from GitHub using

`git clone http://www.github.com/IvoVellekoop/openwfs.git`

The examples are located in the openwfs/examples folder. To build the documentation from the source code and run the automated tests, some additional dependencies are required, which can be installed automatically with [Poetry](https://python-poetry.org/) by running

`poetry install --with dev --with docs`

from the openwfs directory. The tests can now be run by running

`poetry run pytest`

from the openwfs directory.

## Hello wavefront shaping

## Bibliography

<a name="id30"></a>1

I. M. Vellekoop and A. P. Mosk. Focusing coherent light through opaque strongly scattering media. *Opt. Lett.*, 32(16):2309–2311, Aug 2007. [doi:10.1364/OL.32.002309](https://doi.org/10.1364/OL.32.002309).

<a name="id21"></a>2

IM Vellekoop, EG Van Putten, A Lagendijk, and AP Mosk. Demixing light paths inside disordered metamaterials. *Optics express*, 16(1):67–80, 2008.

<a name="id22"></a>3

Dorian Bouchet, Stefan Rotter, and Allard P Mosk. Maximum information states for coherent scattering measurements. *Nature Physics*, 17(5):564–568, 2021.

<a name="id16"></a>4

Antoine Boniface, Mickael Mounaix, Baptiste Blochet, Rafael Piestun, and Sylvain Gigan. Transmission-matrix-based point-spread-function engineering through a complex medium. *Optica*, 4(1):54–59, 2017.

<a name="id19"></a>5

Tomáš Čižmár, Michael Mazilu, and Kishan Dholakia. In situ wavefront correction and its application to micromanipulation. *Nature Photonics*, 4(6):388–394, 2010.

<a name="id18"></a>6

Lorenzo Valzania and Sylvain Gigan. Online learning of the transmission matrix of dynamic scattering media. *Optica*, 10(6):708–716, 2023.

<a name="id17"></a>7

Bahareh Mastiani and Ivo M Vellekoop. Noise-tolerant wavefront shaping in a hadamard basis. *Optics express*, 29(11):17534–17541, 2021.

<a name="id29"></a>8

PS Salter, M Baum, I Alexeev, M Schmidt, and MJ Booth. Exploring the depth range for three-dimensional laser machining with aberration correction. *Optics express*, 22(15):17644–17656, 2014.

<a name="id28"></a>9

Martin Plöschner, Tomáš Tyc, and Tomáš Čižmár. Seeing through chaos in multimode fibres. *Nature Photonics*, 9(8):529–535, 2015.

<a name="id37"></a>10

Abhilash Thendiyammal, Gerwin Osnabrugge, Tom Knop, and Ivo M. Vellekoop. Model-based wavefront shaping microscopy. *Opt. Lett.*, 45(18):5101–5104, Sep 2020. [doi:10.1364/OL.400985](https://doi.org/10.1364/OL.400985).

<a name="id20"></a>11

DWS Cox, T Knop, and IM Vellekoop. Model-based aberration corrected microscopy inside a glass tube. *arXiv preprint arXiv:2311.13363*, 2023.

<a name="id24"></a>12

Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop. Wavefront shaping for forward scattering. *Optics Express*, 30:37436, 10 2022. [doi:10.1364/oe.470194](https://doi.org/10.1364/oe.470194).

<a name="id31"></a>13

Yan Liu, Cheng Ma, Yuecheng Shen, Junhui Shi, and Lihong V. Wang. Focusing light inside dynamic scattering media with millisecond digital optical phase conjugation. *Optica*, 4(2):280–288, Feb 2017. [doi:10.1364/OPTICA.4.000280](https://doi.org/10.1364/OPTICA.4.000280).

<a name="id32"></a>14

Omer Tzang, Eyal Niv, Sakshi Singh, Simon Labouesse, Greg Myatt, and Rafael Piestun. Wavefront shaping in complex media with a 350 khz modulator via a 1d-to-2d transform. *Nature Photonics*, 2019. [doi:10.1038/s41566-019-0503-6](https://doi.org/10.1038/s41566-019-0503-6).

<a name="id33"></a>15

Tomáš Čižmár, Michael Mazilu, and Kishan Dholakia. In situ wavefront correction and its application to micromanipulation. *Nature Photonics*, 4:388–394, 05 2010. [doi:10.1038/nphoton.2010.85](https://doi.org/10.1038/nphoton.2010.85).

<a name="id34"></a>16

Lina Streich, Juan Carlos Boffi, Ling Wang, Khaleel Alhalaseh, Matteo Barbieri, Ronja Rehm, Senthilkumar Deivasigamani, Cornelius T. Gross, Amit Agarwal, and Robert Prevedel. High-resolution structural and functional deep brain imaging using adaptive optics three-photon microscopy. *Nature Methods 2021 18:10*, 18:1253–1258, 9 2021. [doi:10.1038/s41592-021-01257-6](https://doi.org/10.1038/s41592-021-01257-6).
