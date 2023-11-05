# core classes used throughout openwfs
import time
import numpy as np
import astropy.units as u
from typing import Union, List
from concurrent.futures import Future, ThreadPoolExecutor
from astropy.units import Quantity
from abc import ABC, abstractmethod

"""
OpenWFS holds a global state to synchronize measurements. This state can be 
- `moving`. Actuators may be active. No measurements can be made.
- `measuring` Detectors may be active. All actuators must remain static.

If a detector is activated (e.g., by calling `trigger`) when the state is `moving`, 
the framework waits until it is safe to transition to the `measuring` state, and only then sends the trigger.

It is safe to make this transition if the following conditions are met:
    * all detectors are ready to be triggered. This is verified by calling ready
        on all detectors, which returns a future that can be awaited.
    * all motion is almost completed, where 'almost' denotes the latency, 
        `latency = min(d.latency for d in detectors)`
        i.e., the time that passes before the first of the detectors starts responding to the trigger.
        This is verified by calling finished(latency) on all actuators,
        which returns a future that can be awaited.

The transition function simply awaits all futures before making the state transition.

Transition from `measuring` to `moving` is completely analogous, 
swapping detectors and actuators in the description above.

## Example
~~~
### serial implementation
fields = np.zeros((n,))
for n in range(N):
    for p in range(P)
        phase = 2 * np.pi * p/P
        slm.phases = phase
        slm.refresh()         # waits for all measurements to complete (- latency of the slm), and triggers the slm
        f1 = cam1.trigger()   # waits for the image on the slm to stabilize, then triggers the measurement
        f2 = cam2.trigger()   # directly triggers cam2, since we already are in the 'measuring' state
        fields[n] += (f2.result() - f1.result()) * np.exp(-j * phase)   # blocks until frames are read

### pipelined implementation
fields = np.zeros((n,))
def process(n, p, f1, f2):
    fields[n] += (f2 - f1) * np.exp(-j * phase)   # note: no need for 'result()' waiting

for n in range(N):
    for p in range(P)
        phase = 2 * np.pi * p/P
        slm.phases = phase
        slm.refresh()         # waits for all measurements to complete (- latency of the slm), and triggers the slm
        f1 = cam1.trigger()   # waits for the image on the slm to stabilize, then triggers the measurement
        f2 = cam2.trigger()   # directly triggers cam2, since we already are in the 'measuring' state
        Worker.enqueue(process, n, p, f1, f2) # runs processing on a separate thread in order of enqueueing

Worker.wait()  # wait until all processing is done
~~~
"""

DONT_WAIT = Future()
DONT_WAIT.set_result(None)


def _await_results_and_call(fn, *args, **kwargs):
    """Helper function that awaits all futures in the argument list, and then calls function fn"""

    def _await(arg):
        if isinstance(arg, Future):
            return arg.result()
        else:
            return arg  # not a future

    awaited_args = [_await(arg) for arg in args]
    awaited_kwargs = {key: _await(arg) for (key, arg) in kwargs.items()}
    return fn(*awaited_args, **awaited_kwargs)


class Device:
    """Base class for detectors and actuators

    This base class implements the synchronization and switching between 'moving' and 'measuring' states.

    Note:
        When setting a public attributes or property of the device, the Device first waits until all actions are
        finished (wait_finished()).
        For example, if we change the ROI or shutter time of a camera after a call to trigger, the thread blocks
        until the frame grab is complete, and then changes the ROI or shutter time.
    """

    # A thread pool for awaiting detector input, actuator stabilization,
    # or for processing data in a non-deterministic order.
    _workers = ThreadPoolExecutor(thread_name_prefix='Device._workers')

    # Global state: 'moving' or 'measuring'
    moving = False

    # List of all active devices
    devices: "List[Device]" = []

    def __init__(self, latency=0.0 * u.ms):
        self._start_time_ns = 0
        self._latency = latency
        Device.devices.append(self)

    def __del__(self):
        try:
            Device.devices.remove(self)
        except ValueError:
            pass  # happens if constructor fails, then `self` is not in the list

    def __setattr__(self, key, value):
        """Prevents modification of public attributes and properties while the device is busy."""
        if not key.startswith('_'):
            self.wait_finished()
        super().__setattr__(key, value)

    @property
    @abstractmethod
    def is_actuator(self):
        """True for actuators, False for detectors"""
        ...

    def _start(self):
        """Switches the state to 'moving' (for actuators) or 'measuring' (for detectors).

        This function changes the global state to 'moving' or 'measuring' if needed,
        and it may block until this state switch is completed.

        After switching, it returns the current time in the _start_time_ns field.
        """

        if Device.moving != self.is_actuator:
            # a transition from moving/measuring or vice versa is needed
            same_type = [device for device in Device.devices if device.is_actuator == self.is_actuator]
            other_type = [device for device in Device.devices if device.is_actuator != self.is_actuator]

            # compute the minimum latency of same_type
            # for instance, when switching to 'measuring', this number tells us how long it takes before any of the
            # detectors actually starts a measurement.
            # If this is a positive number, we can make the switch to 'measuring' slightly _before_ all actuators have
            # stabilized.
            latency = min((device.latency for device in same_type), default=0.0 * u.ns)

            # wait until all devices of the same kind are ready to be triggered or receive new commands
            for device in same_type:
                device.wait_ready()

            # wait until all devices of the other type have (almost) finished
            for device in other_type:
                device.wait_finished(latency)

        self._start_time_ns = time.time_ns()

    @property
    def latency(self) -> Quantity[u.ms]:
        return self._latency

    def wait_ready(self):
        """Waits until the device is ready to be triggered or take new commands.

        By default, this is the same as wait_finished(self.latency).
        However, some devices may take new commands before they are finished.
        For example, a stage may accept move commands while still moving, updating the target position.
        """
        self.wait_finished(self.latency)

    @abstractmethod
    def wait_finished(self, up_to: Quantity[u.ns] = 0 * u.ns):
        """Waits until the device has finished measuring or 'moving'.

        By default, this is the same as wait_finished(self.latency).
        However, some devices may take new commands before they are finished.
        For example, a stage may accept move commands while still moving, updating the target position.
        """
        ...


class Actuator(Device, ABC):
    """Base class for all actuators
    """

    def is_actuator(self):
        return True


class DataSource(Device, ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior.
    """

    def __init__(self, *, data_shape, pixel_size: Quantity):
        super().__init__()
        self._pixel_size = pixel_size
        self._data_shape = data_shape

    def is_actuator(self):
        return False

    @property
    @abstractmethod
    def measurement_duration(self) -> Quantity[u.ms]:
        """Duration of the measurement (excluding data transfer and latency).

        Returns None if the duration is not known.
        """
        ...

    def wait_finished(self, up_to: Quantity[u.ms] = 0 * u.ms):
        """Waits until the measurement is almost finished (up to 'up_to' nanoseconds)

        If a measurement_duration is defined, this value is used to determine when the measurement is finished.
        Detectors that do not specify a measurement time must override this function.
        """
        assert self.measurement_duration is not None
        end_time = self._start_time_ns + (self.measurement_duration + self.latency).to_value(u.ns)
        time_to_wait = end_time - up_to.to_value(u.ns) - time.time_ns()
        if time_to_wait > 0:
            time.sleep(time_to_wait / 1.0E9)

    def trigger(self, *args, **kwargs) -> Future:
        """Triggers the data source to start acquisition of the data.

        Use await or .result() to wait for the data.
        All parameters are passed to the _fetch function of the detector.
        If any of these parameters is a Future, it is awaited before calling _fetch.
        This way, data from multiple sources can be combined (see Processor).

        Child classes may override trigger() to add additional parameters.
        """
        self._start()
        self._do_trigger()

        def do_fetch(*args_, **kwargs_):
            data = _await_results_and_call(self._fetch, *args_, **kwargs_)
            data.dtype = np.dtype(data.dtype, metadata={'pixel_size': self.pixel_size})
            assert data.shape == self.data_shape
            return data

        return Device._workers.submit(do_fetch, *args, **kwargs)

    def _do_trigger(self):
        """Override to perform the actual hardware trigger"""
        pass

    @abstractmethod
    def _fetch(self, *args, **kwargs) -> np.ndarray:
        """Implement to read the data from the detector

        The args and kwargs are passed from the call to trigger()/
        Note:
            After reading the data, the Detector attaches the pixel_size metadata.
        """
        ...

    def read(self):
        """Triggers the detector and waits for the data to arrive.

        Shortcut for trigger().result().
        """
        return self.trigger().result()

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

    @property
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


class Processor(DataSource, ABC):
    """Helper base class for chaining DataSources.

    Processors can be used to build data processing graphs, where each Processor takes input from one or
    more input DataSources and processes that data (e.g. cropping an image, averaging over an roi, etc.).
    A processor, itself, is a DataSource to allow chaining multiple processors together to combine functionality.

    To implement a processor, override read and/or __init__

    Note: cannot specify latency, it is auto-computed based on the latency of the sources
    """

    def __init__(self, *args, data_shape=None, pixel_size=None):
        self._sources = args
        if data_shape is None:
            data_shape = self._sources[0].data_shape
        if pixel_size is None:
            pixel_size = self._sources[0].pixel_size

        super().__init__(data_shape=data_shape, pixel_size=pixel_size, latency=0 * u.ms)

    def trigger(self, *args, **kwargs):
        """Triggers all sources at the same time (regardless of latency), and schedules a call to `_fetch()`"""
        future_data = [(source.trigger() if source is not None else None) for source in self._sources]
        return super().trigger(*future_data, *args, **kwargs)

    @property
    def latency(self) -> Quantity[u.ms]:
        """Returns the shortest latency for all detectors."""
        return min((source.latency for source in self._sources if source is not None), default=0 * u.ms)

    @property
    def measurement_duration(self) -> Quantity[u.ms]:
        """Returns the longest measurement_duration + latency, minus the shortest latency for all detectors."""
        latency = self.latency
        return max((source.latency + source.measurement_duration for source in self._sources if source is not None),
                   default=latency) - latency


class Worker:
    """The main thread has a single worker thread
    that is used to offload data processing and to prevent the need for blocking to wait for detector data
    in the main thread.

    Note: this is not thread safe, only the main thread should submit work to the worker.
    If we want to make this thread safe, there should be a Worker for each thread that needs one, and
    _executor and _last_task should be thread-local variables.
    """
    """A single (global) thread for processing data in a fixed order."""
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Worker._executor')
    _last_task = DONT_WAIT

    @staticmethod
    def equeue(fn, *args, **kwargs):
        Worker._last_task = Worker._executor.submit(_await_results_and_call, fn, *args, **kwargs)
        return Worker._last_task

    @staticmethod
    def wait():
        """Waits until all previously enqueued work is completed."""
        Worker._last_task.result()


class PhaseSLM(Actuator, ABC):
    phases: np.ndarray

    @abstractmethod
    def update(self):
        """Refresh the SLM to show the updated phase pattern.
        Returns before the image is actually displayed.
        Use the `wait_finished` property to explicitly wait for the update.
        Usually, this is not needed because the framework waits for stabilization of the phase pattern
        before starting a measurement anyway.
        Implementations must call self._start()
        """


def get_pixel_size(data: np.ndarray):
    return data.dtype.metadata['pixel_size']
