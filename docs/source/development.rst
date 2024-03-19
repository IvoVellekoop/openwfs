OpenWFS Development
====================

Running the tests and examples
+++++++++++++++++++++++++++++++

Building the documentation
++++++++++++++++++++++++++++++++++

To build the documentation, make sure to install the required packages first using `poetry install --with docs`. Then, run the following commands to build the html and pdf versions of the documentation, and to auto-generate `README.md`.

.. code-block:: shell

    cd docs
    .\make.bat clean
    .\make.bat html
    .\make.bat markdown
    .\make.bat latex
    cd _build\latex
    xelatex openwfs.tex
    xelatex openwfs.tex


Reporting bugs and contributing
++++++++++++++++++++++++++++++++++



Implementing your own processor
+++++++++++++++++++++++++++++++
To implement a custom processor, derive from the `Processor` base class. Override the `__init__` function to pass all sources to the base class constructor.

In addition, implement the `_fetch` method. The processor will wait until the data from the sources is available, and then call `_fetch` with this data as input arguments. The user should implement this function to process the data and return the result. See `Microscope._fetch`, or any other `Processor` object for an example of how to do this.


Implementing your own detector
+++++++++++++++++++++++++++++++
To implement a detector for a custom device, the user should subclass the `Detector` base class, and implement whatever properties and logic appropriate to the device.

If `duration`, `pixel_size` and `data_shape` are known, they should be passed to the base class constructor. If these properties may change during operation, the user should override the `duration`, `pixel_size` and `data_shape` properties to provide the correct values dynamically. If the `duration` is not known in advance (for example, when waiting for a hardware trigger), the Detector should implement the `busy` function to poll the hardware for the busy state.

To implement triggering the detector and fetching the data, the user should implement the `_do_trigger` and `_fetch` methods. The `_do_trigger` method should start the measurement process. The `_fetch` method should fetch the data from the hardware, process it, and return it as a numpy array.

If the detector was created with the flag `multi_threaded = True`, then `_fetch` will be called from a worker thread. This way, the rest of the program does not need to wait for transferring data from the hardware, or for computationally expensive processing tasks. OpenWFS automatically prevents any modification of public properties between the calls to `_do_trigger` and `_fetch`, which means that the `_fetch` function can safely read (not write) these properties without the chance of a race condition.


Internals of the synchronization mechanism
+++++++++++++++++++++++++++++++++++++++++
For the synchronization, `Device` keeps a global state which can be either
- `moving = True`. One or more actuators may be busy. No measurements can be made (none of the detectors is busy).
- `moving = False` (the 'measuring' state). One or more detectors may be busy. All actuators must remain static (none of the actuators is busy).

When an actuator is started, or when a detector is triggered, the actuator or detector code calls `self._start` to request a switch to the correct global state. If a state switch is needed, this function blocks until all devices of the other type are ready. For example, if a detector calls `_start`, the framework waits for all actuators to finish, before the switch is made.

Whenever possible, an implementation of a `Device` should specify a `duration`, which is the maximum time interval between the call to `_start` and the moment the detector has finished measuring, or the actuator has finished moving. This time is used in the state switching mechanism of `_start`, together with the `latency` of the device. The default implementation of `busy` just returns `True` if a time of at least `duration` has passed since the last call to `_start`. If the device does not know the duration in advance, it should pass an infinite `duration`, and override `busy` to provide a custom implementation.

The `Device` class also provides a `wait` method. This methods blocks until the action of that device is ready. Since the state switching mechanism already automates synchronization, there usually is no need to call `wait` explicitly, except when using the `out` parameter to store measurements in a pre-defined location (see Section Detectors above).

Note that the two use cases of `wait` for detectors are slightly different. When used with the state switching algorithm, it should wait until the hardware has finished acquisition of the data. When used with the `out` parameter, it should wait until the data has been written to the array. To disambiguate between these use cases, the `wait` method has an optional flag `await_data`. When `await_data` is `True`, the method waits until the data has been written to the array. When `await_data` is `False`, the method only waits until the hardware has finished acquisition of the data. The default value of `await_data` is `True`, whereas `_start` internally calls `wait` with `await_data=False`.


 Synchronization:
        Device implements the synchronization between detectors and actuators.
        The idea is that a measurement can only be made when all actuators are stable,
        and that an actuator can only be moved when all detectors are ready.

        The synchronization mechanism is implemented using a global state variable `_moving`.
        A detector can request a state switch to the `measuring` state (`_moving=False`),
        and an actuator can request a state switch to the `moving` state (`_moving=True`)
        by calling `_start`.

        Before making the state switch, `_start` waits for all devices of the _other_ type
        (actuators or detectors) to become ready by calling
        `wait(up_to=latency, await_data=False)` on those devices.
        Here, `latency` is the minimum latency of all devices of the _same_ type as the one
        requesting the state switch.
        If this minimum latency is positive, it means that the devices will not start their
        measurement or movement immediately, so we can make the state switch slightly before
        the devices of the other type are all ready.
        For example, a spatial light modulator has a relatively long latency, meaning that
        we can send the next frame even before a camera has finished reading the previous frame.

        For detectors, `_start` is called automatically by `trigger()`, so there is never a need to call it.
        Implementations of an actuator should call `_start` explicitly before starting to move the actuator.

