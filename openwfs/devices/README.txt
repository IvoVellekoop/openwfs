::In this folder are the examples of devices that can be loaded into MicroManager using PyDevice.

::To test one out without hardware, the files with the 'example' prefix are recommended.

::This folder contains:
- example_camera: A camera object returning noise. Also shows how object inheritance works.
- example_device: A simple device showing all the available types of properties.
- example_xystage: A XYstage object that is registered as an XYstage. For simulated stage-microscope-camera interactions,
    look at openwfs.simulation.microscope for more examples.
- example_zstage: A Zstage object that is registered as a Zstage
- galvo_scanner: Controller for a laser-scanning microscope with two galvo mirrors controlled by a National Instruments
    data acquisition card (nidaq).
- nidaq_gain: A device that controls the voltage of a PMT gain using a NI data acquisition card.