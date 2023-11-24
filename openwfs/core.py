# core classes used throughout openWFS
import time
import atomics
import numpy as np
import astropy.units as u
from weakref import WeakSet
from typing import Union, Set, final
from concurrent.futures import Future, ThreadPoolExecutor
from astropy.units import Quantity
from abc import ABC, abstractmethod


def get_pixel_size(data: np.ndarray):
    """Extracts the `pixel_size` metadata from the data returned from a detector.
    Usage:

     from openwfs.simulation import MockDetector
     det = MockDetector()
     data = det.trigger().result()
     pixel_size = get_pixel_size(data)

    """
    try:
        return data.dtype.metadata['pixel_size']
    except (KeyError, TypeError):
        raise KeyError("data does not have pixel size metadata.")


class Device:
    """Base class for detectors and actuators

    This base class implements the synchronization and switching between 'moving' and 'measuring' states.
    OpenWFS holds a global state to synchronize measurements. This state can be
    - `moving`. Actuators may be active. No measurements can be made.
    - `measuring` Detectors may be active. All actuators must remain static.

    If a detector is activated (e.g., by calling `trigger`) when the state is `moving`,
    the framework waits until it is safe to transition to the `measuring` state, and only then sends the trigger.

    It is safe to make this transition if the following conditions are met:
        1. all detectors are ready to be triggered. This is verified by calling ready
            on all detectors, which returns a future that can be awaited.
        2. all actuators are almost completed, where 'almost' denotes the latency,
            `latency = min(d.latency for d in detectors)`
            i.e., the time that passes before the first of the detectors starts responding to the trigger.
            This is verified by calling finished(latency) on all actuators,
            which returns a future that can be awaited.

    Transition from `measuring` to `moving` is completely analogous,
    swapping detectors and actuators in the description above.

    Examples:
        Serial implementation::

            fields = np.zeros((n,))
            for n in range(N):
                for p in range(P)
                    phase = 2 * np.pi * p / P
                    slm.set_phases(phase) # waits for all measurements to complete (- latency of the slm),
                                          # and then triggers the slm
                    f1 = cam1.trigger()   # waits for the image on the slm to stabilize, then triggers the measurement
                    f2 = cam2.trigger()   # directly triggers cam2, since we already are in the 'measuring' state
                    fields[n] += (f2.result() - f1.result()) * np.exp(-j * phase)   # blocks until frames are read

        Pipelined implementation::

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

    Note:
        When setting a public attributes or property of the device, the Device first waits until all actions
        of that detector are finished (wait_finished()).
        For example, if we change the ROI or shutter time of a camera after a call to trigger, the thread blocks
        until the frame grab is complete, and then changes the ROI or shutter time.
    """

    # A thread pool for awaiting detector input, actuator stabilization,
    # or for processing data in a non-deterministic order.
    _workers = ThreadPoolExecutor(thread_name_prefix='Device._workers')

    # Global state: 'moving'=True or 'measuring'=False
    _moving = False

    # List of all Device objects
    _devices: "Set[Device]" = WeakSet()

    def __init__(self, *, latency=0.0 * u.ms, duration=0.0 * u.ms):
        """Constructs a new Device object

        Args:
            latency: value to use for the `latency` attribute.
                Child classes may directly write to the _latency attribute to modify this value later.

            duration: time it takes to perform the measurement or for the actuator to stabilize.
        """
        self._start_time_ns = 0
        self._latency = latency
        self._duration = duration
        Device._devices.add(self)

    def __del__(self):
        self.wait()

    def __setattr__(self, key, value):
        """Prevents modification of public attributes and properties while the device is busy."""
        if not key.startswith('_'):
            self._wait_finished()
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

        After switching, stores the current time in the _start_time_ns field.
        """

        if Device._moving != self.is_actuator:
            # a transition from moving/measuring or vice versa is needed
            same_type = [device for device in Device._devices if device.is_actuator == self.is_actuator]
            other_type = [device for device in Device._devices if device.is_actuator != self.is_actuator]

            # compute the minimum latency of same_type
            # for instance, when switching to 'measuring', this number tells us how long it takes before any of the
            # detectors actually starts a measurement.
            # If this is a positive number, we can make the switch to 'measuring' slightly _before_ all actuators have
            # stabilized.
            latency = min((device.latency for device in same_type), default=0.0 * u.ns)

            # wait until all devices of the same kind are ready to be triggered or receive new commands
            for device in same_type:
                device._wait_ready()

            # wait until all devices of the other type have (almost) finished
            for device in other_type:
                device._wait_finished(latency)

        self._start_time_ns = time.time_ns()

    @property
    def latency(self) -> Quantity[u.ms]:
        """latency (Quantity[u.ms]): time between sending a command or trigger to the device and the moment the device
            starts responding.
        """
        return self._latency

    @property
    def duration(self) -> Quantity[u.ms]:
        """ duration (Quantity[u.ms]): time it takes to perform the measurement or for the actuator to stabilize.

        This value does not include latency.
        Child classes may directly write to the _duration attribute to modify this value when starting the
        measurement/movement.
        The duration is used in the default implementation of `wait_finished()`.
        If the duration of an operation is not known in advance
        (e.g., waiting for the user to push a button), the child class should pass `np.inf * u.ms` for the
        duration, and override `wait_finished()`.
        """
        return self._duration

    def _wait_ready(self):
        """Waits until the device is ready to be triggered or take new commands.

        By default, this is the same as wait_finished(self.latency).
        However, some devices may take new commands before they are finished.
        For example, a stage may accept move commands while still moving, updating the target position.
        """
        self._wait_finished(self.latency)

    def _wait_finished(self, up_to: Quantity[u.ms] = 0 * u.ns):
        """Waits until the device has (almost) finished measuring or 'moving'.

        This function uses the specified _duration.
        If _duration is np.inf, the wait() function is called instead.

        Args:
            up_to(Quantity[u.ms]): when non-zero, specifies that this function may return 'up_to' milliseconds
                *before* the device is finished.
        """
        if not np.isfinite(self._duration):
            return self.wait()

        end_time = self._start_time_ns + (self._duration + self.latency).to_value(u.ns)
        time_to_wait = end_time - up_to.to_value(u.ns) - time.time_ns()
        if time_to_wait > 0:
            time.sleep(time_to_wait / 1.0E9)

    def wait(self):
        """Waits until the device is completely done with all actions (i.e. busy == False).

        By default, this function just calls self._wait_finished().
        `Detector` overrides this function to wait for the final acquisition to finish.
        `Actuator` objects that do not specify a finite _duration must override this function
            to check if the actuator has finished moving.
        """
        assert np.isfinite(self._duration)
        return self._wait_finished()

    def busy(self) -> bool:
        """Returns true if the device is measuring or moving (see `wait()`)"""
        assert np.isfinite(self._duration)
        end_time = self._start_time_ns + (self._duration + self.latency).to_value(u.ns)
        return end_time > time.time_ns()


class Actuator(Device, ABC):
    """Base class for all actuators
    """

    @final
    def is_actuator(self):
        return True


class Detector(Device, ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior.
    """

    def __init__(self, *, data_shape, pixel_size: Quantity, **kwargs):
        super().__init__(**kwargs)
        ndim = len(data_shape)
        self._pixel_size = pixel_size if pixel_size.size == ndim else np.tile(pixel_size, (ndim,))
        self._data_shape = data_shape
        self._measurements_pending = atomics.atomic(width=4, atype=atomics.INT)
        self._error = None

    @final
    def is_actuator(self):
        return False

    def wait(self):
        """Waits until all measurements are completed.

        Example:
            for i in range(10):
                detector.trigger(out = data[i, ...])
            detector.wait() # wait for the measurements to complete and be stored in `data`
        """
        while self._measurements_pending.load() != 0:
            time.sleep(0.0)
        if self._error is not None:
            e = self._error
            self._error = None
            raise e

    def trigger(self, *args, out=None, immediate=False, **kwargs) -> Future:
        """Triggers the data source to start acquisition of the data.

        Use await or .result() to wait for the data.
        All parameters are passed to the _fetch function of the detector.
        If any of these parameters is a Future, it is awaited before calling _fetch.
        This way, data from multiple sources can be combined (see Processor).

        Child classes may override trigger() to add additional parameters.
        """
        self._start()
        self._do_trigger()
        self._measurements_pending.inc()
        if immediate:
            result = Future()
            result.set_result(self._do_fetch(out, *args, **kwargs))  # noqa
            return result
        else:
            return Device._workers.submit(self._do_fetch, out, *args, **kwargs)

    def _do_fetch(self, out_, *args_, **kwargs_):
        """Helper function that awaits all futures in the keyword argument list, and then calls _fetch"""
        try:
            awaited_args = [(arg.result() if isinstance(arg, Future) else arg) for arg in args_]
            awaited_kwargs = {key: (arg.result() if isinstance(arg, Future) else arg) for (key, arg) in
                              kwargs_.items()}
            data = self._fetch(out_, *awaited_args, **awaited_kwargs)
            data.dtype = np.dtype(data.dtype, metadata={'pixel_size': self.pixel_size})
            assert data.shape == self.data_shape
            return data
        except Exception as e:
            # if we are storing the result in an `out` array,
            # the user may never call result() on the returned future object,
            # and the error may be lost.
            # Therefore, store it so that it can be raised on the next call to wait()
            if out_ is not None:
                self._error = e
            raise e
        finally:
            self._measurements_pending.dec()

    def _do_trigger(self):
        """Override to perform the actual hardware trigger"""
        pass

    @abstractmethod
    def _fetch(self, out: Union[np.ndarray, None], *args, **kwargs) -> np.ndarray:
        """Read the data from the detector

        Args:
            out(ndarray) numpy array or view of an array that will receive the data
                when present, the data will be stored in `out`, and `out` is returned.
                Otherwise, a new array is returned.
                Can also be used with views.
                For example, `trigger(out = data[i, ...])`
            The args and kwargs are passed from the call to trigger()
        Note:
            After reading the data, the Detector attaches the pixel_size metadata.
        Note:
            Child classes must implement this function, and store the data in `out[...]` if `out` is not None.
        """
        ...

    @final
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
            * The property may change, for example, when the ROI of a camera is changed.
              In any case, the value of `data_shape`
              just before calling `trigger` will match the size of the data returned by the corresponding `result` call.
        """
        return self._data_shape

    @property
    def pixel_size(self) -> Quantity:
        """Dimension of one element in the returned data array.

        For cameras, this is the pixel size (in astropy length units).
        For detectors returning a time trace, this value is specified in astropy time units.

        The pixel_size is a 1-D array with an element for each dimension of the returned data.

        By default, the pixel size cannot be set.
        However, in some cases (such as when the `pixel_size` is actually a sampling interval),
              it makes sense for the child class to implement a setter.
        """
        return self._pixel_size

    @final
    def coordinates(self, dim):
        """Coordinate values along the d-th axis.

        The coordinates represent the center of the pixels (or sample intervals),
        so they range from 0.5*pixel_size to (data_shape[d]-0.5) * pixel_size.
        Coordinates have the same astropy base unit as pixel_size.

        The coordinates are returned as an array with the same number of dimensions as the returned data,
        with the d-th dimension holding the coordinates.
        This faclilitates meshgrid-like computations, e.g.
         `cam.coordinates(0) + cam.coordinates(1)` gives a 2-dimensional array of coordinates.

        Args:
            dim: Dimension for which to return the coordinates.

        Returns:
            Quantity: An array holding the coordinates.
        """
        c = np.arange(0.5, 0.5 + self.data_shape[dim], 1.0) * self._pixel_size[dim]
        shape = np.ones(len(self.data_shape), dtype='uint32')
        shape[dim] = self.data_shape[dim]
        return c.reshape(shape)

    @final
    @property
    def extent(self) -> Quantity:
        """Returns the physical size of the data: data_shape * pixel_size.

        Note:
            The value is returned as a numpy array with astropy unit rather than as a tuple to allow easy manipulation.
        """
        return self.data_shape * self.pixel_size


class Processor(Detector, ABC):
    """Helper base class for chaining Detectors.

    Processors can be used to build data processing graphs, where each Processor takes input from one or
    more input Detectors and processes that data (e.g., cropping an image, averaging over an ROI, etc.).
    A processor, itself, is a Detector to allow chaining multiple processors together to combine functionality.

    To implement a processor, override read and/or __init__

    Note: it is not possible to specify the latency, it is computed based on the latency of the sources
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
    def duration(self) -> Quantity[u.ms]:
        """Returns the longest duration + latency, minus the shortest latency for all detectors."""
        latency = self.latency
        return max((source.latency + source.duration for source in self._sources if source is not None),
                   default=latency) - latency


class PhaseSLM(Actuator, ABC):

    @abstractmethod
    def update(self):
        """Sends the new phase pattern to be displayed on the SLM.

        Implementations should call _start() before triggering the SLM.

        Note:
            This function *does not* wait for the image to appear on the SLM.
            To wait for the image stabilization explicitly, use 'wait()'.
            However, this should rarely be needed since all Detectors
            already wait for the image to stabilize before starting a measurement.
        """

    @abstractmethod
    def set_phases(self, values: Union[np.ndarray, float], update=True):
        """Sets the phase pattern on the SLM.

        Args:
            values(ndarray): phase pattern, in radians
            update(bool): when True, calls `update` after setting the phase pattern.
                set to `False` to suppress the call to `update`.
                This is useful in advanced scenarios where multiple parameters of the SLM need to be changed
                before updating the displayed image.
        """
        ...
