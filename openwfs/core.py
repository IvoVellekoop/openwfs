# core classes used throughout openWFS
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Sequence
from typing import Set, final
from weakref import WeakSet

import astropy.units as u
import numpy as np
from astropy.units import Quantity


def set_pixel_size(data: np.ndarray, pixel_size: Quantity) -> np.ndarray:
    data = np.array(data)
    data.dtype = np.dtype(data.dtype, metadata={'pixel_size': pixel_size})
    return data


def get_pixel_size(data: np.ndarray | Quantity, may_fail: bool = False) -> Quantity:
    """Extracts the `pixel_size` metadata from the data returned from a detector.
    Args:
        data(np.ndarray|Quantity): array that holds `pixel_size` metadata, as set by `set_pixel_size`
        may_fail(bool): when set to True, silently set missing pixel size to 1.0 dimensionless
    Usage:

     from openwfs.simulation import MockDetector
     det = MockDetector()
     data = det.trigger().result()
     pixel_size = get_pixel_size(data)
    """
    try:
        return data.dtype.metadata['pixel_size']
    except (KeyError, TypeError):
        if may_fail:
            return 1.0 * u.dimensionless_unscaled
        else:
            raise KeyError("data does not have pixel size metadata.")


def unitless(data):
    """Converts data to a unitless numpy array.

    Args:
        data: If data is a Quantity, convert it to unitless_unscaled (this will raise an error if data has
            a unit attached).
            Otherwise, data is just returned as is.
    """
    if isinstance(data, Quantity):
        return data.to_value(u.dimensionless_unscaled)
    else:
        return data


class Device:
    """Base class for detectors and actuators

    This base class implements the synchronization of detectors and actuators.
    Devices hold a thread lock, which is automatically locked when an attribute is set
    (by overriding __setattr__).
    The lock prevents concurrent modification of the detector settings in a multi-threaded environment.
    In addition, detectors lock the object on `trigger()` and release it after `_fetch`-ing the last measurement data.

    In addition to the locking mechanism, each device can either be `busy` or `ready`.
    Detectors are `busy` as long as the detector hardware is measuring.
    That is, between `trigger()` and the end of the physical measurement,
    which by definition occurs some time _before_ `_fetch` completes,
    so the detector may become `ready` _before_ the measurement data is all fetched and processed.
    Actuators are `busy` when they are moving, about to move, or settling after movement.
    To check if a device is busy, read the `busy` attribute.
    To wait for a device to become `ready`, call `wait`.

    `wait` takes an additional optional time parameter `up_to`.
    When present, the `wait` function will try to return `up_to` seconds before the device becomes ready.
    This feature is used to synchronize detectors and actuators with a latency (see below).

    For detectors, `wait` has an additional flag `await_data`.
    When `True` (default), the function waits until the detector becomes unlocked
    (i.e. also waits for `_fetch` to complete), and `up_to` is ignored.
    When `False`, just waits until the detector has finished the physical measurements, but it may still be
    transferring and processing the data.

    Devices may indicate a `duration`, which is the time between the transition `ready`->`busy`
    (made by an internal call to _start, which is done when triggering a detector, or starting an actuator update)
    and the `busy`->`ready` transition.
    If a `duration` is specified, the `Device` base class takes care of the `busy`->`ready` transition automatically.
    If `duration = 0 * u.ms`, the device is never `busy`.
    This makes sense for data sources that are not physical detectors, such as mock data sources.
    If the measurement/movement duration is nog known in advance,
    devices should set `duration=None` and override `busy` and `wait` to poll if the device is rady.

    OpenWFS synchronizes detectors and actuators using the `ready`->`busy`
    state transition.  If a device needs to switch from `ready` to `busy`, it internally calls `_start`.
    `start` _blocks_ until all devices of the other type are `ready` (up to latency).

    For example, when a detector is triggered, `trigger()` internally calls `_start`,
    which calls `wait(up_to=latency)` on all actuators,
    where `latency` is given by the shortest latency of _all_ detectors.

    For efficiency, Device keeps a global state to synchronize detectors and actuators. This state can be
    - `moving`. Actuators may be active (`busy`). No measurements can be made (all detectors are `ready`)
    - `measuring` Detectors may be active (`busy`). All actuators must remain static (`ready`).
    The `_start` function checks if OpenWFS is already in the correct global state for measuring or moving.
    Only if a state switch is needed, `wait` is called on all objects of the other type, as described above.

    Example:
        ``f1 = np.zeros((N, P, *cam1.data_shape))
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
        fields = (f2 - f1) * np.exp(-j * phase)``
    """

    # A thread pool for awaiting detector input, actuator stabilization,
    # or for processing data in a non-deterministic order.
    _workers = ThreadPoolExecutor(thread_name_prefix='Device._workers')

    # Global state: 'moving'=True or 'measuring'=False
    _moving = False

    # Lock for switching global state (_start)
    _state_lock = threading.Lock()

    # List of all Device objects
    _devices: "Set[Device]" = WeakSet()

    # Option to globally disable multi-threading
    # This is particularly useful for debugging
    multi_threading: bool = True  # False

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
        self._timeout_margin = 5 * u.s
        self._lock = threading.Lock()
        self._locking_thread = None
        self._error = None
        Device._devices.add(self)

    def __del__(self):
        self.wait()

    def __setattr__(self, key, value):
        """Prevents modification of public attributes and properties while the device is busy.

        Private attributes can be set without locking.
        Note that this is not thread-safe and should be done with care!!
        """
        if not key.startswith('_'):
            logging.debug("acquiring lock to set attribute `%s` %s (tid: %i). ", key, self, threading.get_ident())
            new_lock = self._lock_acquire()
            try:
                logging.debug("setting attribute `%s` %s (tid: %i). ", key, self, threading.get_ident())
                super().__setattr__(key, value)
                logging.debug("releasing lock %s (tid: %i).", self, threading.get_ident())
            finally:
                if new_lock:
                    self._lock_release()
        else:
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

        # acquire a global lock, to prevent multiple threads to switch moving/measuring state simultaneously

        # check if transition from moving/measuring or vice versa is needed
        if Device._moving != self.is_actuator():
            if Device._moving:
                logging.debug("switch to MEASURING requested by %s.", self)
            else:
                logging.debug("switch to MOVING requested by %s.", self)

            same_type = [device for device in Device._devices if device.is_actuator == self.is_actuator]
            other_type = [device for device in Device._devices if device.is_actuator != self.is_actuator]

            # compute the minimum latency of same_type
            # for instance, when switching to 'measuring', this number tells us how long it takes before any of the
            # detectors actually starts a measurement.
            # If this is a positive number, we can make the switch to 'measuring' slightly _before_ all actuators have
            # stabilized.
            latency = min((device.latency for device in same_type), default=0.0 * u.ns)

            # wait until all devices of the other type have (almost) finished
            for device in other_type:
                device.wait(up_to=latency, await_data=False)

            # changes the state from moving to measuring and vice versa
            Device._moving = not Device._moving
            if Device._moving:
                logging.debug("state is now MOVING.")
            else:
                logging.debug("state is now MEASURING.")

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

    def wait(self, up_to: Quantity[u.ms] = 0 * u.ns, await_data=True):
        """Waits until the device is (almost) in the `ready` state, i.e. has finished measuring or moving.

        Devices that don't have a fixed duration (_duration = None) should override `busy` to poll for finalization.

        Note:
            In a multi-threading environment,
            there is no guarantee that the device is in the `ready`
            state when this function returns, since some other thread may have activated it again.

        Args:
            up_to(Quantity[u.ms]): when non-zero, specifies that this function may return 'up_to' milliseconds
                *before* the device is finished.
            await_data(bool):
                If False: waits until the device is no longer busy
                If True (default): waits until the device is no longer locked.
                For detectors, this means waiting until all acquisition
                and processing of previously triggered frames is completed.
                If an `out` parameter was specified in `trigger()`, this
                guarantees that the data is stored in the `out` array.
                Since actuators are usually not locked, this flag has no effect.
        """
        if self._error is not None:
            e = self._error
            self._error = None
            raise e

        if self._duration is None or not np.isfinite(self._duration):
            while self.busy():
                time.sleep(0.001)
        else:
            end_time = self._start_time_ns + (self._duration + self.latency).to_value(u.ns)
            time_to_wait = end_time - up_to.to_value(u.ns) - time.time_ns()
            if time_to_wait > 0:
                time.sleep(time_to_wait / 1.0E9)

        if await_data:
            # locks the device, for detectors this waits until all pending measurements are processed.
            if not self._lock_acquire():
                raise RuntimeError(
                    "Cannot call `wait` from inside a setter or from inside _fetch as it will cause a deadlock.")
            self._lock_release()

    def _lock_acquire(self) -> bool:
        """Acquires a non-persistent lock using the timeout of the device"""
        tid = threading.get_ident()
        if self._locking_thread == tid:
            return False  # already locked, this could happen if we call a setter from _fetch or from another setter
        if not self._lock.acquire(timeout=self.timeout.to_value(u.s)):
            raise TimeoutError("Timeout in %s (tid %i)", self, threading.get_ident())
        self._locking_thread = tid
        return True

    def _lock_release(self):
        """Releases a non-persistent lock"""
        self._locking_thread = None
        self._lock.release()

    def busy(self) -> bool:
        """Returns true if the device is measuring or moving (see `wait()`)"""
        assert np.isfinite(self._duration)
        end_time = self._start_time_ns + (self._duration + self.latency).to_value(u.ns)
        return end_time > time.time_ns()

    @property
    def timeout(self) -> Quantity[u.ms]:
        if self._duration is not None and np.isfinite(self._duration):
            return self._timeout_margin + self._duration
        else:
            return self._timeout_margin

    @timeout.setter
    def timeout(self, value):
        if self._duration is not None and np.isfinite(self._duration):
            self._timeout_margin = value - self._duration
        else:
            self._timeout_margin = value.to(u.ms)


class Actuator(Device, ABC):
    """Base class for all actuators
    """

    @final
    def is_actuator(self):
        return True


class Detector(Device, ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior.
    """

    def __init__(self, *, data_shape: Sequence[int], pixel_size: Quantity, **kwargs):
        super().__init__(**kwargs)
        ndim = len(data_shape)
        self._pixel_size = pixel_size if pixel_size.size == ndim else np.tile(pixel_size, (ndim,))
        self._data_shape = tuple(int(x) for x in data_shape)
        self._measurements_pending = 0
        self._pending_count_lock = threading.Lock()
        self._error = None

    @final
    def is_actuator(self):
        return False

    def _lock_persistent(self):
        """Places a persistent lock on the detector.

        A persistent lock is a lock that is only released after all measurements have finished.

        If the detector was not locked, it is locked, and the `measurements_pending` counter is incremented.
        Only when the measurements_pending counter drops to zero, the lock is released (see `_unlock_persistent()`).

        If the detector was already locked, either wait for the lock to be released
        (if the existing lock was not persistent),
        or just increment the `measurements_pending` counter
        (if the existing lock was persistent).
        """
        logging.debug("entering persistent lock %s (tid %i).", self, threading.get_ident())
        with self._pending_count_lock:
            if self._measurements_pending == 0:
                if not self._lock_acquire():  # we don't have a persistent lock yet, acquire the object lock
                    raise RuntimeError("Cannot call `trigger` from a setter or from _fetch")
                self._locking_thread = None
                logging.debug("first persistent lock set %s (tid %i).", self, threading.get_ident())
            self._measurements_pending += 1

    def _unlock_persistent(self):
        """Decrement the `measurements_pending` counter, and release the persistent lock if it reached 0."""
        logging.debug("leaving persistent lock %s (tid %i).", self, threading.get_ident())
        with self._pending_count_lock:
            self._measurements_pending -= 1
            if self._measurements_pending == 0:  # we released the last persistent lock
                self._lock_release()
                logging.debug("last persistent lock cleared %s (tid %i).", self, threading.get_ident())

    def trigger(self, *args, out=None, immediate=False, **kwargs) -> Future:
        """Triggers the detector to start acquisition of the data.

        This function returns a `concurrent.futures.Future`.
        Call `.result()` on the returned object to wait for the data.

        All parameters are passed to the _fetch function of the detector.
        If any of these parameters is a Future, it is awaited before calling _fetch.
        This way, data from multiple sources can be combined (see Processor).

        Child classes may override trigger() to add additional parameters.
        """

        self._lock_persistent()

        # if the detector is not locked yet,
        # lock it now (this is a persistent lock,
        # which is released in _do_fetch if the number of pending measurements reaches zero
        # does nothing if the lock is already locked.
        try:
            self._start()
            assert not Device._moving
            self._do_trigger()
        except:  # noqa  - ok, we are not really catching the exception, just making sure the lock gets released
            self._unlock_persistent()
            raise

        logging.debug("triggering %s (tid: %i).", self, threading.get_ident())
        if immediate or not Device.multi_threading:
            result = Future()
            result.set_result(self._do_fetch(out, *args, **kwargs))  # noqa
            return result
        else:
            return Device._workers.submit(self._do_fetch, out, *args, **kwargs)

    def _do_fetch(self, out_, *args_, **kwargs_):
        """Helper function that awaits all futures in the keyword argument list, and then calls _fetch"""
        try:
            if len(args_) > 0 or len(kwargs_) > 0:
                logging.debug("awaiting inputs for %s (tid: %i).", self, threading.get_ident())
            awaited_args = [(arg.result() if isinstance(arg, Future) else arg) for arg in args_]
            awaited_kwargs = {key: (arg.result() if isinstance(arg, Future) else arg) for (key, arg) in
                              kwargs_.items()}
            logging.debug("fetching data of %s ((tid: %i)).", self, threading.get_ident())
            data = self._fetch(out_, *awaited_args, **awaited_kwargs)
            data = set_pixel_size(data, self.pixel_size)
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
            self._unlock_persistent()

    def _do_trigger(self):
        """Override to perform the actual hardware trigger"""
        pass

    @abstractmethod
    def _fetch(self, out: np.ndarray | None, *args, **kwargs) -> np.ndarray:
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
    def read(self, *args, immediate=True, **kwargs):
        """Triggers the detector and waits for the data to arrive.

        Shortcut for trigger().result().
        """
        return self.trigger(*args, immediate=immediate, **kwargs).result()

    @property
    def data_shape(self):
        """Returns a tuple corresponding to the shape of the returned data array.

        May be overridden by a child class.
        Note:
            This value matches the `shape` property of the array returned when calling `trigger` followed by `read`.
            The property may change, for example, when the ROI of a camera is changed.
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

    def trigger(self, *args, immediate=False, **kwargs):
        """Triggers all sources at the same time (regardless of latency), and schedules a call to `_fetch()`"""
        future_data = [(source.trigger(immediate=immediate) if source is not None else None) for source in
                       self._sources]
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


"""Base class for phase-only SLMs

Attributes:
    extent(Sequence[float]): in a pupil-conjugate configuration, this attribute can be used to 
        indicate how the pattern shown in `set_phases` is mapped to the back pupil of the microscope objective.
        The default extent of (2.0, 2.0) corresponds to exactly filling the back pupil (i.e. the full NA)
        with the phase pattern.
        A higher value can be used to introduce some `bleed`/overfilling to allow for alignment inaccuracies.
"""


class PhaseSLM(Actuator, ABC):
    def __init__(self, extent=np.array((2.0, 2.0))):
        super().__init__()
        self.extent = extent

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
    def set_phases(self, values: np.ndarray | float, update=True):
        """Sets the phase pattern on the SLM.

        Args:
            values(ndarray): phase pattern, in radians
            update(bool): when True, calls `update` after setting the phase pattern.
                set to `False` to suppress the call to `update`.
                This is useful in advanced scenarios where multiple parameters of the SLM need to be changed
                before updating the displayed image.
        """
        ...
