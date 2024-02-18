Detectors, actuators and processors
==================================================

OpenWFS provides a framework for working with detectors and actuators, and for synchronizing their operations.

Detectors
---------
Detectors in OpenWFS are objects that capture, generate, or process data. A Detector object may correspond to a physical device such as a camera, or it may be a software component that generates synthetic data (see :doc:`simulation`).

To acquire data from a detector, call `trigger`. This initiates the process of capturing or generating data and returns a `concurrent.Future` object that will receive the data as it becomes available. The program can continue operation while the data is being captured/transferred/generated in a worker thread. When the data is needed, call `result()` on the `Future` object to wait for the acquisition to complete and retrieve the data.

Here is a typical usage pattern::

    # Trigger the detector, which starts the data capture process
    future = detector.trigger()

    # Do some other work, perhaps trigger other detectors to capture
    # data simultaneously...

    # Now read the data from the detector. If the data is not ready yet,
    # this will block until it is.
    data = future.result()

    # The data is now available for further processing

Sometimes, it is convenient to let the detector store the data at a specified position in an array, rather than explicitly retrieving the data from the `Future`. This can be done by passing an array to the `trigger` method::

    measurements = np.zeros((phase_steps,))
    for p in range(phase_steps):
        self.slm.set_phases(p * 2 * np.pi / phase_steps)
        self.feedback.trigger(out=measurements[p])

    # wait for the last measurement to complete
    measurements.wait()
    # the data is now stored in the `measurements` array


Actuators
---------

Actuators in OpenWFS are objects that perform actions based on the data captured by the detectors. They are typically used to control the state of a system in response to the data captured by the detectors.

Synchronization
---------------


Processors
----------

