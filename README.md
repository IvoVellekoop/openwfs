# OpenWFS
This repository holds Python code for coducting or simulating a wide range of wavefront shaping experiments.
It holds code for different algorithms:
* Stepwise Sequential Algorithm (SSA) [^1].
* Dual Refeference Fourier Algorithm [^2]. 

In addition, it holds code for controlling one or more phase-only spatial light modulators (SLM) connected a graphics card. This code is OpenGL-accelerated and the
SLMs support a variety of functions, such as a disk-shape geometry [^3]. Moreover, the code supports software synchronization between the SLM and a detector, 
and features a pipelined measurement mode that avoids the delay normally caused by the latency of the video card and SLM [^4].

**Note: we are currently in the process of porting our complete code base from MATLAB to Python. The code in this repository is currently _not_ suitable 
for use just yet. A first release of the code is planned November 2023. Please contact us if you are interested in contributing**

[^1]:Vellekoop, I. M., & Mosk, A. P. (2007). Focusing coherent light through opaque strongly scattering media. Optics letters, 32(16), 2309-2311.
[^2]:Mastiani, B., Osnabrugge, G., & Vellekoop, I. M. (2022). Wavefront shaping for forward scattering. Optics express, 30(21), 37436-37445.
[^3]:Mastiani, B., & Vellekoop, I. M. (2021). Noise-tolerant wavefront shaping in a Hadamard basis. Optics express, 29(11), 17534-17541.
[^4};Vellekoop, I. M. (2008). Controlling the propagation of light in disordered scattering media. PhD thesis University of Twente. arXiv preprint arXiv:0807.1087.
