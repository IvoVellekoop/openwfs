OpenWFS
************************************************************
This repository holds Python code for conducting and simulating a wide range
of wavefront shaping experiments in a Python environment.

OpenWFS is a modular platform for:

* Controlling spatial light modulators using OpenGL acceleration.
* Controlling a laser-scanning microscope using a NI Data Acquisition Card.
* Conducting wavefront shaping experiments using a variety of algorithms.
* Simulating wavefront shaping experiments using a variety of algorithms.

Highlighted features:

* automatic synchronization of the SLM and camera, and a pipelined measurement mode that avoids the delay normally caused by the latency of the video card and SLM.
* a modular design that allows for easy addition of new algorithms and devices.
* extensive framework for testing and simulating wavefront shaping algorithms, including the effect of noise, and user-defined aberrations.


Note: this code is designed so that it can be used in conjunction with MicroManager,
a free and open-source microscopy, without any modification.
To use this code in MicroManager, you need the PyDevice plugin, which can be found here:
https://www.github.com/IvoVellekoop/pydevice

Installation instructions
============================================================
>>> pip install openwfs

To run the examples, create a local directory and clone the repository from github using

>>> git clone http://www.github.com/IvoVellekoop/openwfs.git

The examples are located in the openwfs/examples folder.


