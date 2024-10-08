.. _section-micromanager:

OpenWFS in Micro-Manager
==============================================

To smoothly enable end-user interaction with wavefront shaping algorithms, the Micro-Manager device adapter PyDevice was developed :cite:`PyDevice`. A more detailed description can be found in the mmCoreAndDevices source tree :cite:`mmCoreAndDevices`. In essence, PyDevice is Micro-Manager adapter that imports objects from a Python script and integrates them as devices, e.g. a camera or stage. OpenWFS was written in compliance with the templates required for PyDevice, which means OpenWFS cameras, scanners and algorithms can be loaded into Micro-Manager as devices. Examples of this are found in the example gallery :cite:`readthedocsOpenWFS`. Further developments due to this seamless connection can be a dedicated Micro-Manager based wavefront shaping GUI.