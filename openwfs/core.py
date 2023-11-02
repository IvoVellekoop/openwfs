# core classes used throughout openwfs
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from abc import ABC, abstractmethod

"""
# Synchronization
There are three synchronization modes to cover different scenarios.

## Automatic synchronization
Each thread holds a list of actuators and detectors. This list is called a *context*.
The objects that are used in a measurement sequence are added to the context using a `with acquire` statement.
This statement makes sure that the current thread has exclusive access to the objects, 
and registers them in the context of the current thread to enable automatic synchronization.

The context can be in two states:
- `moving`. Signifies that actuators are active, or have just been active.
    During this phase, no measurements can be made. The context starts as 'moving'.
- `measuring` Signifies that detectors are active, or have just been active.
    During this phase, the actuators cannot move.

### Transition to `measuring`
A transition to the `measuring` state occurs when a detector is activated (e.g., by calling `trigger`).
If the context is currently in the `moving` state, wait until the following conditions are met:
    max([a.busy_until for a in context.actuators]) < current_time + min(d.latency for d in context.detectors)
    and all([d.ready for d in context.detectors])

This condition means that all actuators have (almost) finished moving, where 'almost' denotes the latency, i.e., the
where 'almost' denotes the latency, and that all detectors are ready to be triggered.
If the context is already in the `measuring` state, there is no need to wait.


### Transition to `moving`
A transition to the `moving` state occurs when an actuator is activated (e.g., by calling `refresh`, or `move_to`).
If the context is currently in the `measuring` state, wait until the following condition is met:
    max([d.busy_until for d in context.detectors]) < current_time + min(a.latency for a in context.actuators)
    and all([a.ready for a in context.actuators])

This condition means that all detectors have (almost) finished measuring, where 'almost' denotes the latency, 
i.e., the time that passes before the first of the actuators starts responding to a command, and that all
detectors are ready to receive commands.
If the context is already in the `measuring` state, there is no need to wait.


### Timing
All functions that activate a detector or actuator return an awaitable object.
For actuators, this object can be ignored. For detectors, `await` the returned value to obtain the data.
This approach allows triggering multiple detectors at the same time, and waiting for the data at a later moment.

## Example
~~~
for i in range(10):
    slm.refresh()               # waits for all measurements to complete (- latency of the slm), and triggers the slm
    futures_1[i] = cam1.trigger()   # waits for the image on the slm to stabilize, then triggers the measurement
    futures_2[i] = cam2.trigger()   # directly triggers cam2, since we already are in the 'measuring' state
        
for i in range(10):
    frames[i] = futures_1[i].wait() + futures_2[i].wait()
~~~
### Requirements
- derive from DataSource or Actuator (this also implements __enter__ and __leave)
- latency(u.ns) property (default: 0)
- ready(bool) property (default implementation: not busy). Ready to take new commands.
- busy(bool) property (implemented by DataSource and Actuator). Moving/measuring.
- busy_until(u.ns) property (implemented by DataSource and Actuator)
- all activation functions
"""


# Synchronization:
#
# Wait until objects are ready to be triggered (any context) and
# reserve objects for exclusive use (in a multi-threading context):
#
# with cam, slm:
#   ...
#
# Start measurement and wait for result / start movement and wait for stabilization:
# with cam, slm:
#   frame = await cam.refresh()
#   await slm.trigger()
#
# Separate start and wait:
# with stage1, stage2, cam1, cam2:
#   s1 = stage1.move_by(100 * u.um)
#   s2 = stage2.move_by(100 * u.um)
#   await s1
#   await s2
#   future1 = cam1.trigger()
#   future2 = cam2.trigger()
#   frame1 = await future1
#   frame2 = await future2
#
# Make detector depend on actuator

#
# alternative: auto sync
# with stage1, stage2, cam1, cam2:      # keeps list of all active detectors and actuators in thread-local storage
#   for i in range(10):
#     stage1.move_by(100 * u.um)        # waits until all active detectors have finished their measurements
#     stage2.move_by(100 * u.um)        # does not need to wait
#     frames_1[i] = cam1.trigger()      # waits until all active actuators have stabilized
#     frames_2[i] = cam2.trigger()      # does not need to wait
#
# This standard behavior can be overridden using an optional input to 'trigger', move_by, etc so that it does not wait
# move_by and trigger return futures that still can be `await`-ed manually.


class Actuator(ABC):
    """Base class for all actuators

    An actuator has three states:
    * 'ready' - The actuator is in a stable state and can be requested to move
    * 'busy' - The actuator is changing state (e.g., moving, or changing the phase pattern)
    * 'reserved' - The actuator is in a stable state it is required to remain in that state for some time
       because a detector is currently making measurements.

    An actuator has one or more functions that change its state (start moving, trigger a frame change).
    These functions wait until the state is no longer 'reserved', initiate the change, and set the Actuator to
    'busy'.
    After the Actuator is stabilized in the required position, the state is changed to 'ready'

    """

    def __init__(self):
        self._awaitables = []
        self.state = 'ready'

    def __await__(self, reservation=None):
        """Waits until the actuator is in the 'ready' state.

        Detectors should override to return when ready to be triggered.
        Actuators should override to return when the actuation state has stabilized (i.e., stopped moving).
        """
        while len(self._awaitables) > 0:
            item = self._awaitables[0]
            self._awaitables.remove(0)
        if reservation is None:
            self.state = 'ready'
        else:
            self._awaitables.append(reservation)
            self.state = 'reserved'

    def reserve(self, awaitable):
        """Reserves the actuator, preventing any change until the awaitable is fulfilled."""
        self._awaitables.append(awaitable)
        self.state = 'busy'


class DataSource(ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior."""

    def __init__(self, pixel_size: Quantity, data_shape):
        self._pixel_size = pixel_size
        self._data_shape = data_shape

    def buffer_depth(self) -> int:
        """Number of measurements that the detector can hold in its buffer.

        If this value is larger than 1, it means we can call `trigger` that many times before calling `read`.
        """
        return 1

    @abstractmethod
    def trigger(self) -> None:
        """Triggers the data source to start acquisition of the data.

        Note:
            * Calls to `trigger` may be ignored by static data sources.
            * Each call to `trigger` should be followed by a matching call to `read`.
            * If `buffer_depth` is larger than 1, it is possible to trigger multiple measurements and read them later.
              Each measurement is then taken with the value of the settings at the moment `trigger` was called.
            * An implementation should not block during this function call,
              since this will mess up the timing if multiple data sources are chained together.
              If the device is not ready to be triggered yet, an exception should be thrown.
            * Data sources that rely on input from other data sources should trigger those sources.
        """
        ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """Waits for the data to be acquired and returns it as a numpy array

        Note:
            * The `shape` of the returned array matches value the `data_shape` property had when `trigger` was called.
            * Each call to `read` should be preceded by a matching call to `trigger`.
        """
        ...

    @property
    def data_shape(self):
        """Returns a tuple corresponding to the shape of the returned data array.

        May be overridden by a child class.
        Note:
            * This value matches the `shape` property of the array returned when calling `trigger` followed by `read`.
            * The property may change, e.g., when the ROI of a camera is changed.
              In any case, the value of `data_shape`
              just before calling `trigger` will match the size of the data returned by the corresponding `read` call.
        """
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        """Changes the shape of the data. Typically, this function should be overridden by the child class"""
        self._data_shape = value

    @property
    @abstractmethod
    def measurement_time(self) -> Quantity:
        """Acquisition time of the measurement, not including overhead for transferring the data.

        The return value is a Quantity with an astropy time unit.

        Note:
        * This is the time between the call to `trigger` and the moment the measurement is completed.
        * This time is used for synchronizing measurements with actuators such as an SLM or translation stage.
        * Data sources that use data from other data sources should report the longest `measurement_time`
          of these sources (assuming they are all triggered simultaneously).
        * Data sources do not perform a physical measurement should return 0.0
        """
        ...

    @property
    @abstractmethod
    def pixel_size(self) -> Quantity:
        """Dimension of one unit in the returned data array.

        For cameras, this is the pixel size (in astropy length units).
        For detectors returning a time trace, this value is specified in astropy time units.
        Note:
            * At the moment, the pixel_size is a scalar, so for multi-dimensional data sources the pixel_size
              must be the same in all dimensions (only square pixels are supported)
            * By default, the pixel size cannot be set.
              However, in some cases (such as when the `pixel_size` is actually a sampling interval),
              it makes sense for the child class to implement a setter.
        """
        return self._pixel_size

    def coordinates(self, d):
        """Coordinate values along the d-th axis.

        The coordinates represent the center of the pixels (or sample intervals),
        so they range from 0.5*pixel_size to (data_shape[d]-0.5) * pixel_size.
        Coordinates have the same astropy base unit as pixel_size.

        The coordinates are returned as an array along the d-th dimension, facilitating meshgrid-like computations,
        i.e. cam.coordinates(0) + cam.coordinates(1) gives a 2-dimensional array of coordinates.


        """
        coords = np.array(range(self.data_shape[d]), ndmin=len(self.data_shape))
        return (np.moveaxis(coords, -1, d) + 0.5) * self.pixel_size

    def dimensions(self):
        """Returns the physical size of the data: data_shape * pixel_size.

        Note:
            The value is returned as a numpy array with astropy unit rather than as a tuple to allow easy manipulation.
        """
        return np.array(self.data_shape) * self.pixel_size


class Processor(DataSource):
    """Helper base class for chaining DataSources.

    Processors can be used to build data processing graphs, where each Processor takes input from one or
    more input DataSources and processes that data (e.g. cropping an image, averaging over an roi, etc.).
    A processor, itself, is a DataSource to allow chaining multiple processors together to combine functionality.

    To implement a processor, override read and/or __init__
    """

    def __init__(self, source):
        self.source = source

    def trigger(self):
        self.source.trigger()

    @property
    def data_shape(self):
        return self.source.data_shape

    @property
    def measurement_time(self):
        return self.source.measurement_time

    @property
    def pixel_size(self):
        return self.source.pixel_size

    def read(self):
        """The default implementation just returns the source data without modification.

        This supports the usage pattern where the __init__ function of a class chains together multiple processors
        (see MockCamera for an example)
        """
        return self.source.read()


class PhaseSLM(ABC):
    phases: np.ndarray

    def update(self, wait_factor=1.0, wait=True):
        """Refresh the SLM to show the updated phase pattern.

        If the SLM is currently reserved (see `reserve`), this function waits until the reservation is almost (*) over before updating the SLM.
        The SLM waits for the pattern of the SLM to stabilize before returning.

        *) In case of SLMs with an idle time (latency), the image may be sent to the hardware already before the reservation is over, as long as the actual image
        on the SLM is guaranteed not to update until the reservation is over.

        :param wait_factor: time to wait for the image to stabilize. Default = 1.0 should wait for a pre-defined time (the `settle_time`) that guarantees stability
        for most practical cases. Use a higher value to allow for extra stabilization time, or a lower value if you want to trigger a measurement before the SLM is fully stable.

        :param wait: when set to False, do not wait for the image to stabilize but reserve the SLM for this period instead. This can be used to pipeline measurements (see `Feedback`).
        The client code needs to explicitly call `wait` to wait for stabilization of the image.
        """
        pass

    def wait(self):
        """Wait for the SLM to become available. If there are no current reservations, return immediately."""
        pass

    def reserve(self, time: Quantity[u.ms]):
        """Reserve the SLM for a specified amount of time. During this time, the SLM pattern cannot be changed."""
        pass
