# OpenWFS
This repository holds Python code for conducting or simulating a wide range of wavefront shaping experiments in a Python environment and in the MicroManager [^1] environment.

It holds code for different algorithms:
* Stepwise Sequential Algorithm (SSA) [^2].
* Dual Refeference Fourier Algorithm [^3].

It additionally holds the powerful pydevice, which enables any Python class to be used in MicroManager as a device, camera, stage and/or spatial light modulator, in a robust and Pythonic manner, with minimal effort. Please consult the README in its folder for use.

It also holds code for controlling a laser-scanning microscope using a NI Data Acquisition Card, also serving as a camera in MicroManager, allowing for easy but maximal high- and low-level control of the laser-scanning procedure.

In addition, it holds code for controlling one or more phase-only spatial light modulators (SLM) connected to a graphics card. This code is OpenGL-accelerated and the
SLMs support a variety of functions, such as a disk-shape geometry [^4]. Moreover, the code supports software synchronization between the SLM and a detector, 
and features a pipelined measurement mode that avoids the delay normally caused by the latency of the video card and SLM [^5].

**Note: we are currently in the process of porting our complete code base from MATLAB to Python. The code in this repository is currently _not_ suitable 
for use just yet. The first release of the code is planned November 2023. Please contact us if you are interested in contributing**

[^1]:https://micro-manager.org/
[^2]:Vellekoop, I. M., & Mosk, A. P. (2007). Focusing coherent light through opaque strongly scattering media. Optics letters, 32(16), 2309-2311.
[^3]:Mastiani, B., Osnabrugge, G., & Vellekoop, I. M. (2022). Wavefront shaping for forward scattering. Optics express, 30(21), 37436-37445.
[^4]:Mastiani, B., & Vellekoop, I. M. (2021). Noise-tolerant wavefront shaping in a Hadamard basis. Optics express, 29(11), 17534-17541.
[^5};Vellekoop, I. M. (2008). Controlling the propagation of light in disordered scattering media. PhD thesis University of Twente. arXiv preprint arXiv:0807.1087.
