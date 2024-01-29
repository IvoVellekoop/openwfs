OpenWFS
************************************************************
This repository holds Python code for conducting and simulating a wide range
of wavefront shaping experiments in a Python environment.

OpenWFS is a modular platform for:
* Controlling spatial light modulators usen OpenGL acceleration
* Controlling a laser-scanning microscope using a NI Data Acquisition Card
* Conducting wavefront shaping experiments using a variety of algorithms
* Simulating wavefront shaping experiments using a variety of algorithms

Highlighted features:
* automatic synchronization of the SLM and camera, and a pipelined measurement mode that avoids the
delay normally caused by the latency of the video card and SLM.
* a modular design that allows for easy addition of new algorithms and devices.
* extensive framework for testing and simulating wavefront shaping algorithms, including the effect of noise,
  and user-defined aberrations.

Note: this code is designed so that it can be used in conjunction with MicroManager,
a free and open-source microscopy, without any modification.
To use this code in MicroManager, you need the PyDevice plugin, which can be found here:
https://www.github.com/IvoVellekoop/pydevice

Installation instructions
============================================================
#. Install ``PyCharm <https://www.jetbrains.com/pycharm/>``_, Python 3.9 or higher (for example,
   the ``Anaconda distribution <https://www.anaconda.com/download>``_) (Note: currently, OpenWFS does not work with Python
   3.12 because some of the packages it uses do not work with Python 3.12 yet), and a git client, for
   example, ``Git for Windows <https://gitforwindows.org/>``_ or ``GitHub Desktop <https://desktop.github.com/>``_, or any
   Mac/Linux equivalent.

#. Install OpenWFS:
    1. Start Pycharm.
    2. If a project is automatically opened, select Close Project in the hamburger menu to return to the welcome screen
    3. In the welcome screen, select Get from VCS.
    4. In the url field, enter: https://github.com/IvoVellekoop/openwfs.git
    5. In the directory field, enter the location where you want OpenWFS to be installed and press ok.

PyCharm should now download the OpenWFS source files.

#. Configure the Python interpreter
    1. In the top right corner, it says something like ``Python 3.9``, or ``no python found``.
       Click the text and select 'add new interpreter' → 'add local interpreter'
    2. You can click 'inherit site packages' to speed up install.


#. Install required Python packages
    1. You should now have the 'requirements.txt' file open, which displays a warning. Select 'install requirements' in
       the warning bar.


#. Run the examples
    1. Navigate to the 'examples' folder and double-click a python script, for example 'slm_demo.py'
    2. Press the ▷ button, next to 'current file' on the top of the screen.
    3. If you get an error about missing packages, you may have to wait until PyCharm has finished installing all
       packages


