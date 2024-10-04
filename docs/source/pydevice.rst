.. _section-pydevice:

OpenWFS in PyDevice
==============================================

To smoothly enable end-user interaction with wavefront shaping algorithms, the Micro-Manager device adapter PyDevice was developed :cite:`MMpydevice`. A more detailed description can be found in the mmCoreAndDevices source tree :cite:`mmCoreAndDevices`. In essence, PyDevice is a c++ based adapter that imports objects from a Python script and integrates them in Micro-Manager as devices, e.g. a camera or stage. OpenWFS was written in compliance with the templates required for PyDevice, which means OpenWFS cameras, scanners and algorithms can be loaded into Micro-Manager as devices. Examples of this are found in the example gallery :cite:`ExampleGallery`. 