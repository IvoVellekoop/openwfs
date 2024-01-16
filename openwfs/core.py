import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Set, final, Sequence, Tuple, Optional, Union
from weakref import WeakSet

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike
from astropy.units import Quantity

# Aliases for commonly used type hints

# An extent is a sequence of two floats with an optional unit attached,
# or a single float with optional unit attached, which is broadcasted to a sequence of two floats.
ExtentType = Union[float, Sequence[float], np.ndarray, Quantity]


def set_pixel_size(data: ArrayLike, pixel_size: Optional[Quantity]) -> np.ndarray:
    """
    Sets the pixel size metadata for the given data array.

    Args:
        data (ArrayLike): The input data array.
        pixel_size (Optional[Quantity]): The pixel size to be set. When a single-element pixel size is given,
            it is broadcasted to all dimensions of the data array.
            Passing None sets the pixel size metadata to None.

    Returns:
        np.ndarray: The modified data array with the pixel size metadata.

    Usage:
    >>> data = np.array([[1, 2], [3, 4]])
    >>> pixel_size = 0.1 * u.m
    >>> modified_data = set_pixel_size(data, pixel_size)
    """
    data = np.array(data)

    if pixel_size is not None and pixel_size.size == 1:
        pixel_size = pixel_size * np.ones(data.ndim)

    data.dtype = np.dtype(data.dtype, metadata={'pixel_size': pixel_size})
    return data


def get_pixel_size(data: np.ndarray) -> Optional[Quantity]:
    """
    Extracts the pixel size metadata from the data array.

    Args:
        data (np.ndarray): The input data array or Quantity.

    Returns:
        OptionalQuantity]: The pixel size metadata, or None if no pixel size metadata is present.

    Usage:
    >>> import astropy.units as u
    >>> import numpy as np
    >>> data = set_pixel_size(((1, 2), (3, 4)), 5 * u.um)
    >>> pixel_size = get_pixel_size(data)
    """
    metadata = data.dtype.metadata
    if metadata is None:
        return None
    return data.dtype.metadata.get('pixel_size', None)


def unitless(data: ArrayLike) -> ArrayLike:
    """
    Converts unitless `Quanity` objects to numpy arrays.

    Args:
        data: The input data.
        If `data` is a Quantity, it is converted to a (unitless) numpy array.
        All other data types are just returned as is.

    Returns:
        ArrayLike: unitless numpy array, or the input data if it is not a Quantity.

    Raises:
        UnitConversionError: If the data is a Quantity with a unit

    Note:
        Do NOT use `np.array(data)` to convert a Quantity to a numpy array,
        because this will drop the unit prefix.
        For example, ```np.array(1 * u.s / u.ms) == 1```.

    Usage:
    >>> data = np.array([1.0, 2.0, 3.0]) * u.m
    >>> unitless_data = unitless(data)
    """
    if isinstance(data, Quantity):
        return data.to_value(u.dimensionless_unscaled)
    else:
        return data


class Device:
    """Base class for detectors and actuators

    Multi-threading:
        Devices hold a thread lock, prevents concurrent modification of attributes
        in a multithreaded environment.
        This lock is automatically acquired when setting a public attribute or property.
        See `__setattr__` for details.

    Detector locking:
        For detectors, an additional synchonization mechanism is implemented.
        It prevents modification of the detector settings while a measurement is in progress.
        See Detector for details.

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

    Usage:
        >>> f1 = np.zeros((N, P, *cam1.data_shape))
        >>> f2 = np.zeros((N, P, *cam2.data_shape))
        >>> for n in range(N):
        >>>     for p in range(P)
        >>>         phase = 2 * np.pi * p / P
        >>>
        >>>         # wait for all measurements to complete (up to the latency of the slm), and trigger the slm.
        >>>         slm.set_phases(phase)
        >>>
        >>>         # wait for the image on the slm to stabilize, then trigger the measurement.
        >>>         cam1.trigger(out = f1[n, p, ...])
        >>>
        >>>         # directly trigger cam2, since we already are in the 'measuring' state.
        >>>         cam2.trigger(out = f2[n, p, ...])
        >>>
        >>> cam1.wait() # wait until camera 1 is done grabbing frames
        >>> cam2.wait() # wait until camera 2 is done grabbing frames
        >>> fields = (f2 - f1) * np.exp(-j * phase)

    Attributes:
    multi_threading (bool): Option to globally disable multi-threading.
        This is particularly useful for debugging.
    _workers (ThreadPoolExecutor): A thread pool for awaiting detector input, actuator stabilization,
       or for processing data in a non-deterministic order.
    _moving (bool): Global state: 'moving'=True or 'measuring'=False
    _state_lock (Lock): Lock for switching global state (see _start)
    _devices (WeakSet[Device]): List of all Device objects
    """
    _workers = ThreadPoolExecutor(thread_name_prefix='Device._workers')
    _moving = False
    _state_lock = threading.Lock()
    _devices: "Set[Device]" = WeakSet()
    multi_threading: bool = True

    def __init__(self):
        """Constructs a new Device object"""
        self._end_time_ns = 0
        self._timeout_margin = 5 * u.s
        self._lock = threading.Lock()
        self._locking_thread = None
        self._error = None
        self._base_initialized = True
        Device._devices.add(self)

    def __del__(self):
        """
        Destructor for Device objects.

        The destructor calls `wait()` to ensure that the device is not busy when it is destroyed.
        If `__init__` did not complete, the destructor does nothing.

        Note:
            When closing Python, all modules are unloaded in an undefined order.
            This can cause problems if `wait` calls a function form a module that is already unloaded, such as `numpy`.
            In this case, a 'NoneType is not callable' error occurs.
        """
        if hasattr(self, '_base_initialized'):
            self.wait()

    def __setattr__(self, key, value):
        """Prevents modification of public attributes and properties while the device is locked.

        For detectors, this prevents modification of the detector settings while a measurement is in progress.
        For all devices, it prevents concurrent modification in a multi-threading context.

        Private attributes can be set without locking.
        Note that this is not thread-safe and should be done with care!!
        """
        if hasattr(self, '_base_initialized') and not key.startswith('_'):
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

        After switching, stores the time at which the operation will have ended in the `_end_time_ns`
        field (i.e., `time.time_ns() + self.latency + self.duration`).
        """

        # acquire a global lock, to prevent multiple threads to switch moving/measuring state simultaneously
        with Device._state_lock:
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
                # If this is a positive number, we can make the switch to 'measuring' slightly _before_
                # all actuators have stabilized.
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

            # also store the time we expect the operation to finish
            # note: it may finish slightly earlier since (latency + duration) is a maximum value
            self._end_time_ns = time.time_ns() + self.latency.to_value(u.ns) + self.duration.to_value(u.ns)

    @property
    def latency(self) -> Quantity[u.ms]:
        """latency (Quantity[u.ms]): minimum amount of time between sending a command or trigger to the device and the
        moment the device starts responding.

        The default value is 0.0 ms.

        Note:
            The latency is used to compute when it is safe to switch the global state from 'moving'
            to 'measuring' or vice versa.
            Devices that report a non-zero latency promise not to do anything before the latency has passed.
            This allows making the state switch even before the devices of the other type have all finished.

            This construct is used for spatial light modulators, which typically have a long latency (1-2 frames).
            Due to this latency, we may send an `update` command to the SLM even before the camera has finished reading
            the previous frame.

        Note:
            A device is allowed to report a different latency every time `latency` is called.
            For example, for a spatial light modulator that only refreshes
            at a fixed rate, we can add the remaining time until the next refresh to the latency.
        """
        return 0.0 * u.ms

    @property
    @abstractmethod
    def duration(self) -> Quantity[u.ms]:
        """duration (Quantity[u.ms]): maximum amount of time it takes to perform the measurement or for the
        actuator to stabilize.
        This value *does not* include the latency.

        For a detector, this is the maximum amount of time that elapses between returning from `trigger()` and the
        end of the measurement.
        For an actuator, this is the maximum amount of time that elapses from returning from a command like `update(
        )` and the stabilization of the device.

        If the duration of an operation is not known in advance,
        (e.g., when waiting for a hardware trigger), this function should return `np.inf`.

        Note: A device may update the duration dynamically.
        For example, a stage may compute the required time to
        move to the target position and update the duration accordingly.

        Note: If `latency` is a (lower) estimate, the duration should be high enough to guarantee that `latency +
        duration` is at least as large as the time between starting the operation and finishing it.
        """
        ...

    def wait(self, up_to: Quantity[u.ms] = 0 * u.ns, await_data=True):
        """Waits until the device is (almost) in the `ready` state, i.e., has finished measuring or moving.

        This function is called by `_start` automatically to ensure proper synchronization between detectors and
        actuators, and it is called by `__del__` to ensure the device is not active when it is destroyed.
        The only time to call `wait` explicitly is when using pipelined measurements, see `Detector.trigger()`.

        For devices that report a duration (`duration ≠ ∞`), this function waits until
        `current_time - up_to >= self._end_time_ns`,
        where `_end_time_ns` was set by the last call to `_start`.

        For devices that report no duration `duration = ∞`, this function repeatedly calls `busy` until `busy`
        returns `False`.
        In this case, `up_to` is ignored.

        Args:
            up_to(Quantity[u.ms]):
                when non-zero, specifies that this function may return 'up_to' milliseconds
                *before* the device is finished.
            await_data(bool):
                If True, after waiting until the device is no longer busy, briefly locks the device.
                For detectors, this has the effect of waiting until all acquisitions and processing of
                previously triggered frames are completed.
                If an `out` parameter was specified in `trigger()`, this guarantees that the data is stored
                in the `out` array.
                For actuators, this flag has no effect other than briefly locking the device.

        Raises:
            Any other exception raised by the device in another thread (e.g., during `_fetch`).
            TimeoutError: if the device has `duration = ∞`, and `busy` does not return `True` within
                `self.timeout`
            RuntimeError: if `wait` is called from inside a setter or from inside `_fetch`.
                This would cause a deadlock.

        """
        if self._error is not None:
            e = self._error
            self._error = None
            raise e

        # If duration = ∞, poll busy until it returns False or a timeout occurs.
        # Note: avoid np.isinf because numpy may have been unloaded during Python shutdown
        if self._end_time_ns > 1.0e+38:
            start = time.time_ns()
            timeout = self.timeout.to_value(u.ns)
            while self.busy():
                time.sleep(0.001)
                if time.time_ns() - start > timeout:
                    raise TimeoutError("Timeout in %s (tid %i)", self, threading.get_ident())
        else:
            time_to_wait = self._end_time_ns - up_to.to_value(u.ns) - time.time_ns()
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
            raise TimeoutError("Timeout in %s (tid %i)", self, tid)
        self._locking_thread = tid
        return True

    def _lock_release(self):
        """Releases a non-persistent lock"""
        self._locking_thread = None
        self._lock.release()

    def busy(self) -> bool:
        """Returns true if the device is measuring or moving (see `wait()`).

        Note: if a device does not define a finite `duration`, it must override this function to poll for
        finalization."""
        return time.time_ns() < self._end_time_ns

    @property
    def timeout(self) -> Quantity[u.ms]:
        """timeout (Quantity[u.ms]): time after which a timeout error is raised when waiting for the device.

        The timeout is automatically adjusted if the `duration` changes.
        The default value is `duration + 5 s`."""
        duration = self.duration
        if not np.isinf(duration):
            return self._timeout_margin + duration
        else:
            return self._timeout_margin

    @timeout.setter
    def timeout(self, value):
        duration = self.duration
        if not np.isinf(duration):
            self._timeout_margin = value - duration
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

    def __init__(self):
        super().__init__()
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

        Child classes may override trigger() to call `super().trigger()` with additional parameters.

        Note:
            To implement hardware triggering, do not override this function.
            Instead, override `_do_trigger()` instead to ensure proper synchronization and locking.
        """

        self._lock_persistent()

        # if the detector is not locked yet,
        # lock it now (this is a persistent lock),
        # which is released in _do_fetch if the number of pending measurements reaches zero
        # does nothing if the lock is already locked.
        try:
            self._start()
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
            raise e  # raise the error again, it will be stored in the 'Future' object that was returned by trigger()
        finally:
            self._unlock_persistent()

    def _do_trigger(self):
        """Override this function to perform the actual hardware trigger."""
        pass

    @abstractmethod
    def _fetch(self, out: Optional[np.ndarray], *args, **kwargs) -> np.ndarray:
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
    def read(self, *args, immediate=True, **kwargs) -> np.ndarray:
        """Triggers the detector and waits for the data to arrive.

        Shortcut for trigger().result().
        """
        return self.trigger(*args, immediate=immediate, **kwargs).result()

    @property
    @abstractmethod
    def data_shape(self) -> Tuple[int, ...]:
        """Returns a tuple corresponding to the shape of the returned data array.

        Must be implemented by the child class.
        Note:
            This value matches the `shape` property of the array returned when calling `trigger` followed by `read`.
            The property may change, for example, when the ROI of a camera is changed.
            In any case, the value of `data_shape`
            just before calling `trigger` must match the size of the data returned by the corresponding `result` call.
        """
        ...

    @property
    def pixel_size(self) -> Optional[Quantity]:
        """Dimension of one element in the returned data array.

        For cameras, this is the pixel size (in astropy length units).
        For detectors returning a time trace, this value is specified in astropy time units.

        The pixel_size is a 1-D array with an element for each dimension of the returned data.

        By default, the pixel size cannot be set.
        However, in some cases (such as when the `pixel_size` is actually a sampling interval),
            it makes sense for the child class to implement a setter.
        """
        return None

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
        c = np.arange(0.5, 0.5 + self.data_shape[dim], 1.0)
        if self.pixel_size is not None:
            c = c * self.pixel_size[dim]
        shape = np.ones_like(self.data_shape)
        shape[dim] = self.data_shape[dim]
        return c.reshape(shape)

    @final
    @property
    def extent(self) -> Quantity:
        """Returns the physical size of the data: data_shape * pixel_size.

        Note:
            The value is returned as a numpy array with astropy unit rather than as a tuple to allow easy manipulation.
            If `self.pixel_size is None`, just returns `self.data_shape`.
        """
        return self.data_shape * self.pixel_size if self.pixel_size is not None else self.data_shape


class Processor(Detector, ABC):
    """Helper base class for chaining Detectors.

    Processors can be used to build data processing graphs, where each Processor takes input from one or
    more input Detectors and processes that data (e.g., cropping an image, averaging over an ROI, etc.).
    A processor, itself, is a Detector to allow chaining multiple processors together to combine functionality.

    To implement a processor, implement _fetch, and optionally override data_shape, pixel_size, and __init__.

    The `latency` and `duration` properties are computed from the latency and duration of the inputs and cannot be set.
    By default, the `pixel_size` and `data_shape` are the same as the pixel_size and data_shape of the first input.
    To override this behavior, override the `pixel_size` and `data_shape` properties.
    """

    def __init__(self, *args):
        self._sources = args
        super().__init__()

    def trigger(self, *args, immediate=False, **kwargs):
        """Triggers all sources at the same time (regardless of latency), and schedules a call to `_fetch()`"""
        future_data = [(source.trigger(immediate=immediate) if source is not None else None) for source in
                       self._sources]
        return super().trigger(*future_data, *args, **kwargs)

    @final
    @property
    def latency(self) -> Quantity[u.ms]:
        """Returns the shortest latency for all detectors."""
        return min((source.latency for source in self._sources if source is not None), default=0.0 * u.ms)

    @final
    @property
    def duration(self) -> Quantity[u.ms]:
        """Returns the last end time minus the first start time for all detectors
        i.e., max (duration + latency) - min(latency).

        Note that `latency` is allowed to vary over time for devices that can only be triggered periodically,
        so this `duration` may also vary over time.
        """
        times = [(source.duration, source.latency) for source in self._sources if source is not None]
        if len(times) == 0:
            return 0.0 * u.ms
        return (max([duration + latency for (duration, latency) in times])
                - min([latency for (duration, latency) in times]))

    @property
    def data_shape(self):
        """This default implementation returns the data shape of the first source."""
        return self._sources[0].data_shape

    @property
    def pixel_size(self) -> Optional[Quantity]:
        """This default implementation returns the pixel size of the first source."""
        return self._sources[0].pixel_size


class PhaseSLM(Actuator, ABC):
    """Base class for phase-only SLMs

    Attributes:
        extent(Sequence[float]): in a pupil-conjugate configuration, this attribute can be used to
            indicate how the pattern shown in `set_phases` is mapped to the back pupil of the microscope objective.
            The default extent of (2.0, 2.0) corresponds to exactly filling the back pupil (i.e., the full NA)
            with the phase pattern.
            A higher value can be used to introduce some `bleed`/overfilling to allow for alignment inaccuracies.
    """

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
    def set_phases(self, values: ArrayLike, update=True):
        """Sets the phase pattern on the SLM.

        Args:
            values(ArrayLike): phase pattern, in radians.
                The pattern is automatically stretched to fill the full SLM.
            update(bool): when True, calls `update` after setting the phase pattern.
                Set to `False` to suppress the call to `update`.
                This is useful in advanced scenarios where multiple parameters of the SLM need to be changed
                before updating the displayed image.
        """
        ...
