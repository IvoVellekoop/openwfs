# OpenWFS

This repository holds Python code for conducting or simulating a wide range of wavefront shaping experiments in a Python
environment and in the MicroManager [^1] environment.

It holds code for different algorithms:

* Stepwise Sequential Algorithm (SSA) [^2].
* Dual Reference Fourier Algorithm [^3].

It additionally holds the powerful pydevice, which enables any Python class to be used in MicroManager as a device,
camera, stage and/or spatial light modulator, in a robust and Pythonic manner, with minimal effort. Please consult the
README in its folder for use.

It also holds code for controlling a laser-scanning microscope using a NI Data Acquisition Card, also serving as a
camera in MicroManager, allowing for easy but maximal high- and low-level control of the laser-scanning procedure.

In addition, it holds code for controlling one or more phase-only spatial light modulators (SLM) connected to a graphics
card. This code is OpenGL-accelerated, and the
SLMs support a variety of functions, such as a disk-shape geometry [^4]. Moreover, the code supports software
synchronization between the SLM and a detector,
and features a pipelined measurement mode that avoids the delay normally caused by the latency of the video card and
SLM [^5].

**Note: we are currently in the process of porting our complete code base from MATLAB to Python. The code in this
repository is currently _not_ suitable
for use just yet. The first release of the code is planned November 2023. Please contact us if you are interested in
contributing**

## Installation instructions

1. Install [PyCharm](https://www.jetbrains.com/pycharm/), Python 3.9 or higher (for example,
   the [Anaconda distribution](https://www.anaconda.com/download)) (Note: currently, OpenWFS does not work with Python
   3.12 because some of the packages it uses do not work with Python 3.12 yet), and a git client, for
   example, [Git for Windows](https://gitforwindows.org/) or [GitHub Desktop](https://desktop.github.com/), or any
   Mac/Linux equivalent.

2. Install OpenWFS:
    1. Start Pycharm.
    2. If a project is automatically opened, select Close Project in the hamburger menu to return to the welcome screen
    3. In the welcome screen, select Get from VCS.
    4. In the url field, enter: https://github.com/IvoVellekoop/openwfs.git
    5. In the directory field, enter the location where you want OpenWFS to be installed and press ok.

PyCharm should now download the OpenWFS source files.

1. Configure the Python interpreter
    1. In the top right corner, it says something like `Python 3.9`, or `no python found`.
       Click the text and select 'add new interpreter' → 'add local interpreter'
    2. You can click 'inherit site packages' to speed up install.


1. Install required Python packages
    1. You should now have the 'requirements.txt' file open, which displays a warning. Select 'install requirements' in
       the warning bar.


1. Run the examples
    1. Navigate to the 'examples' folder and double-click a python script, for example 'slm_demo.py'
    2. Press the ▷ button, next to 'current file' on the top of the screen.
    3. If you get an error about missing packages, you may have to wait until PyCharm has finished installing all
       packages

[^1]:https://micro-manager.org/
[^2]:Vellekoop, I. M., & Mosk, A. P. (2007). Focusing coherent light through opaque strongly scattering media. Optics
letters, 32(16), 2309-2311.
[^3]:Mastiani, B., Osnabrugge, G., & Vellekoop, I. M. (2022). Wavefront shaping for forward scattering. Optics express,
30(21), 37436-37445.
[^4]:Mastiani, B., & Vellekoop, I. M. (2021). Noise-tolerant wavefront shaping in a Hadamard basis. Optics express, 29(
11), 17534-17541.
[^5]:Vellekoop, I. M. (2008). Controlling the propagation of light in disordered scattering media. PhD thesis University
of Twente. arXiv preprint arXiv:0807.1087.
