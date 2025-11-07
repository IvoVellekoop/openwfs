.. _section-micromanager:

OpenWFS in Micro-Manager
==============================================

For end users of microscopes, it is important to have a user-friendly interface to control the wavefront shaping algorithms. For this reason, we developed an adapter for the Micro-Manager is a widely-used open-source microscopy control software package Micro-Manger :cite:`micromanager,MMoverview`. Our adapter, called PyDevice :cite:`PyDevice`, allows any Python object to be loaded in the Micro-Manager graphical user interface (GUI). Objects that can be loaded include cameras, stages, wavefront shaping algorithms, and generic Python objects that expose attributes to the GUI. PyDevice allows the user to control the wavefront shaping algorithms from the Micro-Manager GUI, and to see feedback on the performance of the algorithms. Examples of this can be found in the example gallery :cite:`readthedocsOpenWFS`. At the time of writing, PyDevice is included as an experimental feature in the Micro-Manager nightly builds. A more detailed description of PyDevice can be found in the mmCoreAndDevices source tree :cite:`mmCoreAndDevices`.

An example of this integration can be found in the online example ``micro_manager_microscope.py``. In this file, a simulated microscope is set up using the OpenWFS simulation tools. In order to expose this to PyDevice, a :class:`dict` object is created named ``devices`` containing the OpenWFS objects from the simulated microscope. These objects can then be used as an ordinary camera and translation stage from Micro-Manager, as seen in figure :numref:`micromanagerconnection`.


.. _micromanagerconnection:
.. figure:: mm_example.png
    :align: center

    The OpenWFS microscope object loaded into Micro-Manager.
    
As any Python object can be exposed to Micro-Manager, this approach can also be used to control wavefront shaping algorithms from the GUI. The :class:`WFSController` object in :class:`openwfs.algorithms.utilities` was made for this purpose. It allows the user to load any algorithm into Micro-Manager, and adjust settings from the GUI. Additionally, feedback on the performance of WFS algorithms (see :numref:`section-troubleshooting`) is exposed, allowing the user to see parameters such as estimated enhancement and signal-to-noise ratio, and toggle the SLM to show the flat or optimized wavefront. An example of how to control a wavefront shaping algorithm in a fully simulated microscopy environment can be found in ``micro_manager_wfs.py`` in the example gallery.
    
