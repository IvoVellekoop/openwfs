Key concepts
==================================================
The OpenWFS framework is built around the concept of *devices*. A device is a piece of hardware or software that can be controlled and/or read out by the framework. Devices can be *detectors*, which capture, process, or synthesize data, or *actuators*, which perform actions based on the data captured by the detectors. The framework provides a common interface for working with detectors and actuators, and for synchronizing their operations.

OpenWFS is not designed to be thread-safe, and the user is responsible for guaranteeing that devices are only accessed from a single thread at a time. For detectors, however, OpenWFS does provide threading support by the means of a worker thread that captures and/or processes data without blocking the main thread, as discussed in the section on :ref:`Asynchronous measurements`.

In addition, OpenWFS maintains metadata and units for all data arrays and properties where relevant. This approach reduce the chance of errors caused by passing a quantity in incorrect units, and to simplify the computation of coordinates (see :ref:`Units and metadata`).


Detectors
------------
Detectors in OpenWFS are objects that capture, generate, or process data. All detectors derive from the :class:`~.Detector` base class. A Detector object may correspond to a physical device such as a camera, or it may be a software component that generates synthetic data (see :ref:`Simulations`). Detectors have the following properties and methods:

.. code-block:: python

    class Detector(Device):

        # starting a measurement
        def read(self) -> np.ndarray
        def trigger(self, out: Optional[np.ndarray] = None) -> Future

        # metadata
        data_shape: Tuple[int, int]
        pixel_size: Optional[Quantity]
        def coordinates(dimension: int) -> Quantity
        extent: Quantity


The :meth:`~.Detector.read()` method of a detector starts a measurement and returns the captured data. It triggers the detector and blocks until the data is available. Data is always returned as `numpy` array. Subclasses of :class:`~.Detector` typically add a set of properties specific to that detector (e.g. shutter time, gain, etc.). In the simplest case, setting these properties and calling :meth:`.~Detector.read()` is all that is needed to capture data. As described in section :ref:`Asynchronous measurements`, the :meth:`~.Detector.trigger()` method is used for asynchronous measurements. All other properties and methods are used for metadata and units, as described in section :ref:`Units and metadata`.

The detector object inherits some properties and methods from the base class :class:`~.Device`. These are used by the synchronization mechanism to determine when it is safe to start a measurement, as described in section :ref:`Synchronization`.

Units and metadata
+++++++++++++++++++++++++++
OpenWFS consistently uses `astropy.units` for quantities with physical dimensions, which allows for calculations to be performed with correct units, and for automatic unit conversion where necessary. Importantly, it prevents errors caused by passing a quantity in incorrect units, such as passing a wavelength in micrometers when the function expects a wavelength in nanometers. Units are converted automatically, so one may for example specify a time in microseconds, milliseconds, or seconds. The use of units is illustrated in the following snippet:

.. code-block:: python

    import astropy.units as u
    c = Camera()
    c.shutter_time = 10 * u.ms
    c.shutter_time = 0.01 * u.s  # equivalent to the previous line
    c.shutter_time = 10 # raises an error, since the unit is missing

In addition, OpenWFS allows attaching coordinate metadata to `numpy` arrays using the functions :func:`~.set_pixel_size()`. Pixel sizes can represent a physical length (e.g. as in the size pixels on an image sensor), or other units such as time (e.g. as the sampling period in a time series). OpenWFS fully supports anisotropic pixels, where the pixel sizes in the x and y directions are different (but still need to have the same base unit).

Each :class:`~.Detector` has a :attr:`~.Detector.pixel_size` property, which is an `astropy.units.Quantity` vector with a pixel size (including unit of measure) for each of the dimensions of the returned data. The pixel size can be obtained from this property, or by calling :func:`~.get_pixel_size()` on the returned data returned by :meth:`~.Detector.read()`.  For detectors that return time signals, the `pixel_size` should be specified in an astropy time unit. If the pixel size is not known, or not meaningful (for example the :class:`.SingleROI` detector outputs a single value with no unit), :attr:`~.Detector.pixel_size` can be set to None.

As an alternative accessing the pixel size directly, :func:`~get_extent()` provide access to the extent of the array, which is always equal to the pixel size times the shape of the array. This metadata is present on the data array returned by the detector, and it can also be obtained from the :class:`~.Detector.extent` property

Finally, detectors have a convenience function for computing coordinate ranges with the appropriate unit. The :meth:`~.Detector.coordinates` method returns a vector of coordinates along a dimension of the array, with the origin at the center of the array. For example, if an array has pixel size 1.0 and shape (3,), the `extent` of the array is 3.0, and the coordinates will be `[-1, 0, 1]`, where the pixels range from -1.5 to -0.5, -0.5 to 0.5, and 0.5 to 1.5, respectively.

Asynchronous measurements
+++++++++++++++++++++++++++
:meth:`.~Detector.read()` blocks the program until the captured data is available. This behavior is not ideal when multiple detectors are used simultaneously, or when transferring or processing the data takes a long time. In these cases, it is preferable to use :meth:`.~Detector.trigger()`, which initiates the process of capturing or generating data and returns a `concurrent.futures.Future` object that will receive the data as it becomes available. The program can continue operation while the data is being captured/transferred/generated in a worker thread. When the data is needed, call `result()` on the `Future` object to wait for the acquisition to complete and retrieve the data.

Here is a typical usage pattern::

.. code-block:: python

    # Trigger the detector, which starts the data capture process
    future = detector.trigger()

    # Do some other work, perhaps trigger other detectors to capture
    # data simultaneously...

    # Now read the data from the detector. If the data is not ready yet,
    # this will block until it is.
    data = future.result()

    # The data is now available for further processing


A third method for obtaining data is to pass a numpy array or view as `out` parameter to :meth:`.~Detector.trigger`. When the data becomes available it is stored in this array automatically. Before processing the data, the user should call :meth:`.~Detector.wait` (see section :ref:`Synchronization`) on the detector object to make sure all data is fetched and safely stored in the output array. For example::

.. code-block:: python

    measurements = np.zeros((phase_steps,))
    for p in range(phase_steps):
        slm.set_phases(p * 2 * np.pi / phase_steps)
        detector.trigger(out=measurements[p])

    # wait for the last measurement to complete
    detector.wait()
    # the data is now stored in the `measurements` array

While fetching and processing data is underway, any attempt to modify a property of the detector will block until the fetching and processing is complete. This way, all properties (such as the region of interest) are guaranteed to be constant between the calls to :meth:`.~Detector.trigger` and the moment the data is actually fetched and processed in the worker thread.


Processors
------------
A `Processor` is a `Detector` that takes input from one or more other detectors, and combines/processes this data to produce its own output. As an example, the `SingleRoiProcessor` allows averaging 2-dimensional image data over a specific region of interest, using a square, disk, or Gaussian mask for weighted averaging. For example, by combining a `Camera` and a `SingleRoiProcessor`, we can make an effective point detector  that produces the average value of several pixels in the camera frame as an output. Since a processor, itself, is a `Detector`, multiple processors can be chained together to combine their functionality. The `Processor` automatically triggers all sources, and awaits their data before processing it.

The OpenWFS further includes various processors, such as a `CropProcessor` to crop data to a rectangular region of interest, and a `TransformProcessor` to perform affine image transformations to image produced by a source. The testing and simulation framework in addition has an `ADCProcessor` to convert the data to integers, while adding optional shot noise and readout noise and saturation to mimic an analog to digital converter.


Actuators
---------
Actuators in OpenWFS are objects that perform actions based on the data captured by the detectors. They are typically used to control the state of a system in response to the data captured by the detectors.
OpenWFS provides a framework for working with detectors and actuators, and for synchronizing their operations. Detectors are devices that *measure* a signal of some kind. Actuators are devices that *move* things in the setup. This can be literal, such as moving a translation stage, or a virtual movement, like an SLM that takes time to switch to a different phase pattern. All detectors and actuators derive from the common :class:`.Device` base class, which provides synchronization between detectors and actuators.


Synchronization
---------------
When running an experiment, it is essential to synchronize detectors and actuators. For example, starting an acquisition on a camera while the spatial light modulator (SLM) is still switching to a new phase pattern will result in an incorrect measurement. Similarly, moving a translation stage while the camera is still acquiring data will result in a blurred image. OpenWFS provides fully automatic synchronization between different devices, so that there never is any need for manual synchronization code (like `sleep` statements).

The :class:`~.Device` base class implements a set of properties and methods to implement the synchronization mechanism:

.. code-block:: python

    class Device:
        def busy(self) -> bool
        def wait(self, await_data: bool = True)

        duration: Quantity[u.ms]
        latency: Quantity[u.ms]
        timeout: Quantity[u.ms]


Each device can either be *busy* or *ready*, and this state can be polled by calling :meth:`~.Device.busy()`. Detectors are busy as long as the detector hardware is measuring.  Actuators are busy when they are moving, about to move, or settling after movement. OpenWFS automatically enforces two conditions:

- before starting a measurement, wait until all motion is (almost) completed
- before starting any movement, wait until all measurements are (almost) completed

Here, 'almost' refers to the fact that devices may have a *latency*. Latency is the time between sending a command to a device, and the moment the device starts responding. An important example is the SLM, which typically takes one or two frame periods to transfer the image data to the liquid crystal chip. Such devices can specify a non-zero `latency` attribute. When specified, the device 'promises' not to do anything until `latency` milliseconds after the start of the measurement or movement. When a latency is specified, detectors or actuators can be started slightly before the devices of the other type (actuators or detectors, respectively) have finished their operation. For example, this mechanism allows sending a new frame to the SLM *before* the measurements of the current frame are finished, since it is known that the SLM will not respond for `latency` milliseconds anyway. This way, measurements and SLM updates can be pipelined to maximize the number of measurements that can be done in a certain amount of time. To enable this pipelined measurements, the `Device` class also provides a `duration` attribute, which is the maximum time interval between triggering the detector or starting the actuator, and the moment the detector has finished measuring, or the actuator has finished moving.

This synchronization is performed automatically and it is usually not necessary write any synchronization code (like `sleep` statements). If desired, it is possible to explicitly wait for the device to become ready by calling :meth:`~.Device.wait()`. Typically, this is only necessary when using the `out` parameter to store measurements in a pre-defined location (see section :ref:`Asynchronous measurements` above). A typical usage pattern is illustrated in the following example:

.. code-block:: python

    f1 = np.zeros((N, P, *cam1.data_shape))
    f2 = np.zeros((N, P, *cam2.data_shape))
    for n in range(N):
        for p in range(P)
            phase = 2 * np.pi * p / P
            # wait for all measurements to complete (up to the latency of the slm), and trigger the slm.
            slm.set_phases(phase)
            # wait for the image on the slm to stabilize, then trigger the measurement.
            cam1.trigger(out = f1[n, p, ...])
            # directly trigger cam2, since we already are in the 'measuring' state.
            cam2.trigger(out = f2[n, p, ...])
    cam1.wait() # wait until camera 1 is done grabbing frames
    cam2.wait() # wait until camera 2 is done grabbing frames
    fields = (f2 - f1) * np.exp(-j * phase)

Finally, devices have a `timeout` attribute, which is the maximum time to wait for a device to become ready. This timeout is used in the state-switching mechanism, and when explicitly waiting for results using :meth:`~.Device.wait()` or  :meth:`~.Device.read()` or by calling `result()` on the `Future` object returned by :meth:`~.Device.trigger()`.


