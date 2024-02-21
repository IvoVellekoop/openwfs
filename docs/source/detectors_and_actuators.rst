Detectors, actuators and processors
==================================================

OpenWFS provides a framework for working with detectors and actuators, and for synchronizing their operations. Detectors are devices that _measure_ a signal of some kind. Actuators are devices that _move_ things in the setup. This can be literal, such as moving a translation stage, or a virtual movement, like an SLM that takes time to switch to a different phase pattern.

Detectors
---------
Detectors in OpenWFS are objects that capture, generate, or process data. A Detector object may correspond to a physical device such as a camera, or it may be a software component that generates synthetic data (see :doc:`simulation`).

Data is always returned as numpy array, with a `pixel_size` metadata attached whenever possible (see :doc:`coordinates`). The `data_shape` and `pixel_size` properties of the detector return the shape (as in the numpy `shape` attribute) and the pixel size of the returned data. The pixel size is an `astropy.units.Quantity`, vector with a pixel size (including unit of measure) for each of the dimensions of the returned data. Anisotropic pixel sizes (i.e. a size that is different for the _x_ and _y_ direction) are fully supported. For detectors that return time signals, the `pixel_size` should be specified in an astropy time unit. If the pixel size is not known, or not meaningful, a value of `None` is used. Shape and pixel size use the convention that the first index in an array corresponds to the vertical coordinate, and the second index to the horizontal coordinate, i.e. `(y, x)`, `(height, width)`. This convention is consistent with the way matplotlib and most other plotting libraries that display a numpy array.


The `read` method of a detector starts a measurement and returns the captured data. It triggers the detector and blocks until the data is available. This behavior is not ideal when multiple detectors are used simultaneously, or when transferring or processing the data takes a long time. In these cases, it is preferable to use `trigger`, which initiates the process of capturing or generating data and returns a `concurrent.futures.Future` object that will receive the data as it becomes available. The program can continue operation while the data is being captured/transferred/generated in a worker thread. When the data is needed, call `result()` on the `Future` object to wait for the acquisition to complete and retrieve the data.

Here is a typical usage pattern::

    # Trigger the detector, which starts the data capture process
    future = detector.trigger()

    # Do some other work, perhaps trigger other detectors to capture
    # data simultaneously...

    # Now read the data from the detector. If the data is not ready yet,
    # this will block until it is.
    data = future.result()

    # The data is now available for further processing


A third method for obtaining data is to pass a numpy array or view as `out` parameter to `trigger`. When the data becomes available it is stored in this array automatically. Before processing the data, the user should call `wait` on the detector object to make sure all data is fetched and safely stored in the output array. For example::

    measurements = np.zeros((phase_steps,))
    for p in range(phase_steps):
        self.slm.set_phases(p * 2 * np.pi / phase_steps)
        self.feedback.trigger(out=measurements[p])

    # wait for the last measurement to complete
    measurements.wait()
    # the data is now stored in the `measurements` array


While fetching and processing data is underway, any attempt to modify a property of the detector will block until the fetching and processing is complete. This way, all properties (such as the region of interest) are guaranteed to be constant between the calls to `trigger` and `\_fetch`.

Actuators
---------

Actuators in OpenWFS are objects that perform actions based on the data captured by the detectors. They are typically used to control the state of a system in response to the data captured by the detectors.


Processors
------------
A `Processor` is a `Detector` that takes input from one or more other detectors, and combines/processes this data to produce its own output. As an example, the `SingleRoiProcessor` allows averaging 2-dimensional image data over a specific region of interest, using a square, disk, or Gaussian mask for weighted averaging. For example, by combining a `Camera` and a `SingleRoiProcessor`, we can make an effective point detector  that produces the average value of several pixels in the camera frame as an output. Since a processor, itself, is a `Detector`, multiple processors can be chained together to combine their functionality. The `Processor` automatically triggers all sources, and awaits their data before processing it.

The OpenWFS further includes various processors, such as a `CropProcessor` to crop data to a rectangular region of interest, and a `TransformProcessor` to perform affine image transformations to image produced by a source. The testing and simulation framework in addition has an `ADCProcessor` to convert the data to integers, while adding optional shot noise and readout noise and saturation to mimic an analog to digital converter.


Synchronization
---------------
OpenWFS provides fully automatic synchronization between different devices. Each device can either be *busy* or *ready*, and this state can be polled by calling `busy`. Detectors are busy as long as the detector hardware is measuring.  Actuators are busy when they are moving, about to move, or settling after movement. The OpenWFS automatically enforces two conditions:

- before starting a measurement, wait until all motion is (almost) completed
- before starting any movement, wait until all measurements are (almost) completed

Here, 'almost' refers to the fact that devices may have a *latency*. Latency is the time between sending a command to a device, and the moment the device starts responding. An important example is the SLM, which typically takes one or two frame periods to transfer the image data to the liquid crystal chip. Such devices can specify a non-zero `latency` attribute. When specified, the device 'promises' not to do anything until `latency` milliseconds after the start of the measurement or movement. When a latency is specified, detectors or actuators can be started slightly before the devices of the other type (actuators or detectors, respectively) have finished their operation. For example, this mechanism allows sending a new frame to the SLM *before* the measurements of the current frame are finished, since it is known that the SLM will not respond for `latency` milliseconds anyway. This way, measurements and SLM updates can be pipelined to maximize the number of measurements that can be done in a certain amount of time.

This synchronization is performed automatically and it is usually not necessary write any synchronization code (like `sleep` statements). The only exception is the call to `wait` when using the `out` parameter to store measurements in a pre-defined location (see Section Detectors above).

..
    # Also see Section~\ref{sec:algorithms} and \cite{ThesisVellekoop}.



Implementation
------------------
For the synchronization, `Device` keeps a global state which can be either
- `moving = True`. One or more actuators may be busy. No measurements can be made (none of the detectors is busy).
- `moving = False` (the 'measuring' state). One or more detectors may be busy. All actuators must remain static (none of the actuators is busy).

When an actuator is started, or when a detector is triggered, the actuator or detector code calls `self._start` to request a switch to the correct global state. If a state switch is needed, this function blocks until all devices of the other type are ready. For example, if a detector calls `_start`, the framework waits for all actuators to finish, before the switch is made.

Whenever possible, an implementation of a `Device` should specify a `duration`, which is the maximum time interval between the call to `_start` and the moment the detector has finished measuring, or the actuator has finished moving. This time is used in the state switching mechanism of `_start`, together with the `latency` of the device. The default implementation of `busy` just returns `True` if a time of at least `duration` has passed since the last call to `_start`. If the device does not know the duration in advance, it should pass an infinite `duration`, and override `busy` to provide a custom implementation.

The `Device` class also provides a `wait` method. This methods blocks until the action of that device is ready. Since the state switching mechanism already automates synchronization, there usually is no need to call `wait` explicitly, except when using the `out` parameter to store measurements in a pre-defined location (see Section Detectors above).

Note that the two use cases of `wait` for detectors are slightly different. When used with the state switching algorithm, it should wait until the hardware has finished acquisition of the data. When used with the `out` parameter, it should wait until the data has been written to the array. To disambiguate between these use cases, the `wait` method has an optional flag `await_data`. When `await_data` is `True`, the method waits until the data has been written to the array. When `await_data` is `False`, the method only waits until the hardware has finished acquisition of the data. The default value of `await_data` is `True`, whereas `_start` internally calls `wait` with `await_data=False`.

To implement a Dectector, Actuator or Processor, the user should subclass the `Device` base class, and implement whatever properties and logic appropriate to the device.

For detectors, if `duration`, `pixel_size` and `data_shape` are known, they should be passed to the base class constructor. If these properties may change during operation, the user should override the `duration`, `pixel_size` and `data_shape` properties to provide the correct values. If the `duration` is not known in advance (for example, when waiting for a hardware trigger), the Detector should implement the `busy` function to poll the hardware for the busy state.

For detectors, the user should implement the `_fetch` method, which should fetch the data from the hardware, process it, and return it as a numpy array. If the detector was created with the flag `multi_threaded = True`, then `_fetch` will be called from a worker thread. This way, the rest of the program does not need to wait for transferring data from the hardware, or for computationally expensive processing tasks. Care must be taken not to modify any properties of the detector in `_fetch`. When trying to write to a public property, an error occurs because all write access to public properties is blocked while one or more measurements are being processed. When writing to a private property, no error occurs, but race conditions may occur. Therefore, the user should make sure that no properties are modified while `_fetch`

For processors, the `_fetch` method after the data from all sources is available. This data is passed as function arguments to `_fetch`, so that it can be processed there. See `Microscope._fetch` for an example of how to do this.

