.. _section-micromanager:

OpenWFS in Micro-Manager
==============================================

To smoothly enable end-user interaction with wavefront shaping algorithms, the Micro-Manager device adapter PyDevice was developed :cite:`PyDevice`. Micro-Manager is a widely-used open-source microscopy control software package. In essence, PyDevice is a Micro-Manager adapter that imports objects from a Python script and integrates them as devices, e.g. a camera or stage. OpenWFS was written in compliance with the templates required for PyDevice, which means OpenWFS cameras, scanners and algorithms can be loaded into Micro-Manager as devices. Examples of this are found in the example gallery :cite:`readthedocsOpenWFS`, and a more detailed description of PyDevice can be found in the mmCoreAndDevices source tree :cite:`mmCoreAndDevices`.

Simulated microscope in Micro-Manager
----------------------------------------------------
An example of this integration can be found in :class:`micro_manager_microscope.py`. In this file, a simulated microscope is set up using the tools in  :class:`openwfs.simulation`. In order to expose this to PyDevice, a :class:`dict` object is created named :class:`devices` containing the OpenWFS objects from the simulated microscope. These objects can then be manipulated from Micro-Manager, as seen in figure :numref:`micromanagerconnection`.


.. _micromanagerconnection:
.. figure:: MMexample.PNG
    :align: center

    The OpenWFS microscope object loaded into Micro-Manager.
    
Beyond hardware components, the algorithms can also be controlled from Micro-Manager, as any Python object can be exposed. The :class:`WFSController` object in :class:`openwfs.algorithms.utilities` was made for this purpose. It allows the user to choose any algorithm, load it into Micro-Manager, and adjust settings from the GUI. Additionally, feedback on the performance of WFS algorithms is exposed, allowing the user to see parameters such as estimated enhancement and signal-to-noise ratio, and toggle the SLM to show the :class:`FLAT` or :class:`OPTIMIZED` wavefront. An example of this can be found in :class:`micro_manager_wfs.py` in the example gallery.
    
