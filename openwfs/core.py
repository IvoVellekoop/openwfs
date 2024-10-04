import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Set, final, Tuple, Optional
from weakref import WeakSet

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike

from .utilities import set_pixel_size


class Device(ABC):
    """Base class for detectors and actuators

    See :ref:`key_concepts` for more information.

    """

    __slots__ = (
        "_end_time_ns",
        "_timeout_margin",
        "_locking_thread",
        "_error",
        "__weakref__",
        "_latency",
        "_duration",
        "_multi_threaded",
    )
    _workers = ThreadPoolExecutor(thread_name_prefix="Device._workers")
    _moving = False
    _state_lock = threading.Lock()
    _devices: "Set[Device]" = WeakSet()

    def __init__(self, *, duration: Quantity[u.ms], latency: Quantity[u.ms]):
        """Constructs a new Device object"""
        self._latency = latency
        self._duration = duration
        self._end_time_ns = 0
        self._timeout_margin = 5 * u.s
        self._locking_thread = None
        self._error = None
        Device._devices.add(self)

    @property
    @abstractmethod
    def _is_actuator(self):
        """True for actuators, False for detectors"""
        ...

    def _start(self):
        """Switches the state to 'moving' (for actuators) or 'measuring' (for detectors).

        This function changes the global state to 'moving' or 'measuring' if needed,
        and it may block until this state switch is completed.

        After switching, stores the time at which the operation will have ended in the ``_end_time_ns``
        field (i.e., ``time.time_ns() + self.latency + self.duration``).
        """

        # acquire a global lock, to prevent multiple threads to switch moving/measuring state simultaneously
        with Device._state_lock:
            # check if transition from moving/measuring or vice versa is needed
            if Device._moving != self._is_actuator():
                if Device._moving:
                    logging.debug("switch to MEASURING requested by %s.", self)
                else:
                    logging.debug("switch to MOVING requested by %s.", self)

                same_type = [device for device in Device._devices if device._is_actuator == self._is_actuator]
                other_type = [device for device in Device._devices if device._is_actuator != self._is_actuator]

                # compute the minimum latency of same_type
                # for instance, when switching to 'measuring', this number tells us how long it takes before any of the
                # detectors actually starts a measurement.
                # If this is a positive number, we can make the switch to 'measuring' slightly _before_
                # all actuators have stabilized.
                latency = min([device.latency for device in same_type], default=0.0 * u.ns)  # noqa - incorrect warning

                # wait until all devices of the other type have (almost) finished
                for device in other_type:
                    device.wait(up_to=latency)

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
        return self._latency

    @property
    def duration(self) -> Quantity[u.ms]:
        """duration (Quantity[u.ms]): maximum amount of time it takes to perform the measurement or for the
        actuator to stabilize.
        This value *does not* include the latency.

        For a detector, this is the maximum amount of time that elapses between returning from `trigger()` and the
        end of the measurement.
        For an actuator, this is the maximum amount of time that elapses from returning from a command like `update(
        )` and the stabilization of the device.

        If the duration of an operation is not known in advance,
        (e.g., when waiting for a hardware trigger), this function should return `np.inf * u.ms`.

        Note: A device may update the duration dynamically.
        For example, a stage may compute the required time to
        move to the target position and update the duration accordingly.

        Note: If `latency` is a (lower) estimate, the duration should be high enough to guarantee that `latency +
        duration` is at least as large as the time between starting the operation and finishing it.
        """
        return self._duration

    def wait(self, up_to: Optional[Quantity[u.ms]] = None) -> None:
        """Waits until the device is (almost) in the `ready` state, i.e., has finished measuring or moving.

        This function is called by `_start` automatically to ensure proper synchronization between detectors and
        actuators, and it is called by `__del__` to ensure the device is not active when it is destroyed.
        The only time to call `wait` explicitly is when using pipelined measurements, see `Detector.trigger()`.

        For devices that report a duration (`duration ≠ ∞`), this function waits until
        `current_time - up_to >= self._end_time_ns`,
        where `_end_time_ns` was set by the last call to `_start`.

        For devices that report no duration `duration = ∞`, this function repeatedly calls `busy` until `busy`
        returns `False`. In this case, `up_to` is ignored.

        Args:
            up_to(Quantity[u.ms]):
                when specified, specifies that this function may return 'up_to' milliseconds
                *before* the device is finished.

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
        if np.isfinite(self._end_time_ns):
            start = time.time_ns()
            timeout = self.timeout.to_value(u.ns)
            while self.busy():
                time.sleep(0.01)
                if time.time_ns() - start > timeout:
                    raise TimeoutError("Timeout in %s (tid %i)", self, threading.get_ident())
        else:
            time_to_wait = self._end_time_ns - time.time_ns()
            if up_to is not None:
                time_to_wait -= up_to.to_value(u.ns)
            if time_to_wait > 0:
                time.sleep(time_to_wait / 1.0e9)

    def busy(self) -> bool:
        """Returns true if the device is measuring or moving (see `wait()`).

        Note: if a device does not define a finite `duration`, it must override this function to poll for
        finalization."""
        return time.time_ns() < self._end_time_ns

    @property
    def timeout(self) -> Quantity[u.ms]:
        """Time after which a timeout error is raised when waiting for the device.

        The timeout is automatically adjusted if the `duration` changes.
        The default value is `duration + 5 s`."""
        duration = self.duration
        if np.isfinite(duration):
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
    """Base class for all actuators"""

    __slots__ = ()

    @final
    def _is_actuator(self):
        return True


class Detector(Device, ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior.

    See :numref:`Detectors` in the documentation for more information.
    """

    __slots__ = (
        "_measurements_pending",
        "_lock_condition",
        "_pixel_size",
        "_data_shape",
    )

    def __init__(
        self,
        *,
        data_shape: Optional[tuple[int, ...]],
        pixel_size: Optional[Quantity],
        duration: Optional[Quantity[u.ms]],
        latency: Optional[Quantity[u.ms]],
        multi_threaded: bool = True
    ):
        """
        Constructor for the Detector class.

        Args:
            data_shape:  The shape of the data array that `read()` will return. When None is passed,
                the subclass should override the `data_shape` property to return the actual shape.
            pixel_size:  The pixel size (in astropy length units). None if the pixels do not have a size.
                Subclassed can override the `pixel_size` property to return the actual pixel size.
            duration: The maximum amount of time that elapses between returning from `trigger()`
                and the end of the measurement. If the duration of an operation is not known in advance,
                (e.g., when waiting for a hardware trigger), this value should be `np.inf * u.ms`
                and the `busy` method should be overridden to return `False` when the measurement is finished.
                If None is passed, the subclass should override the `duration` property to return the actual duration.
            latency: The minimum amount of time between sending a command or trigger to the device
                and the moment the device starts responding. If None is passed, the subclass should override
                the `latency` property to return the actual latency.
            multi_threaded: If True, `_fetch` is called from a worker thread. Otherwise, `_fetch` is called
                directly from `trigger`. If the device is not thread-safe, or threading provides no benefit,
                or for easy debugging, set this to False.
        """
        super().__init__(duration=duration, latency=latency)
        self._measurements_pending = 0
        self._lock_condition = threading.Condition()
        self._error = None
        self._data_shape = data_shape
        self._pixel_size = pixel_size
        self._multi_threaded = multi_threaded

    @final
    def _is_actuator(self):
        return False

    def _increase_measurements_pending(self):
        with self._lock_condition:
            self._measurements_pending += 1

    def _decrease_measurements_pending(self):
        with self._lock_condition:
            self._measurements_pending -= 1
            if self._measurements_pending == 0:
                self._lock_condition.notify_all()

    def wait(self, up_to: Quantity[u.ms] = None) -> None:
        """Waits until the hardware has (almost) finished measuring

        Due to the automatic synchronization between detectors and actuators, this function only needs to be called
        explicitly when waiting for data to be stored in the `out` argument of :meth:`~.Detector.trigger()`.

        Args:
            up_to: if specified, this function may return `up_to` milliseconds *before* the hardware
                has finished measurements.
                If None, this function waits until the hardware has finished all measurements *and* all data is fetched,
                and stored in the `out` array if that was passed to trigger().

        """
        super().wait(up_to)
        if up_to is None:
            # wait until all pending measurements are processed.
            with self._lock_condition:
                while self._measurements_pending > 0:
                    self._lock_condition.wait()

    def trigger(self, *args, out=None, immediate=False, **kwargs) -> Future:
        """Triggers the detector to start acquisition of the data.

        This function does not wait for the measurement to complete.
        Instead, it returns a ``concurrent.futures.Future``.
        Call ``.result()`` on the returned object to wait for the data.
        Here is a typical usage pattern:

        .. code-block:: python

            # Trigger the detector, which starts the data capture process
            future = detector.trigger()

            # Do some other work, perhaps trigger other detectors to capture
            # data simultaneously...

            # Now read the data from the detector. If the data is not ready yet,
            # this will block until it is.
            data = future.result()

        An alternative method for asynchronous data capture is to use
        the `out` parameter to specify a location where to store the data:

        .. code-block:: python

            out = np.zeros((2,), dtype='float32')
            detector.trigger(out=out[0])  # start the first measurement
            detector.trigger(out=out[1])  # queue the second measurement
            detector.wait()  # wait for both measurements to complete
            # Now the data is stored in the `out` array.

        All input parameters are passed to the _fetch function of the detector.
        Child classes may override trigger() to call `super().trigger()` with additional parameters.
        If any of these parameters is a Future, it is awaited before calling _fetch.
        This way, data from multiple sources can be combined (see Processor).

        Note:
            To implement hardware triggering, do not override this function.
            Instead, override `_do_trigger()` to ensure proper synchronization and locking.

        Args:
            out: If specified, the data is stored in this array once it is available.
            immediate: If True, the data is fetched in the current thread. This is useful for debugging,
                and for cases where the data is needed immediately. It avoids the overhead (and debugging complications)
                of dispatching the call to _fetch to a worker thread.
            *args: Additional arguments passed to the _fetch function.
                If any of these arguments is a `concurrent.futures.Future`, the data is awaited before calling _fetch,
                and the data is passed instead of the `Future`.
                This is useful for combining data from multiple sources (see `Processor`).
            **kwargs: Additional keyword arguments passed to the _fetch function. Any `concurrent.futures.Future`
                in the keyword arguments is awaited before calling _fetch,
                and the data is passed instead of the `Future`.
        """
        self._increase_measurements_pending()
        try:
            self._start()
            self._do_trigger()
        except:  # noqa  - ok, we are not really catching the exception, just making sure the lock gets released
            self._decrease_measurements_pending()
            raise

        logging.debug("triggering %s (tid: %i).", self, threading.get_ident())
        if immediate or not self._multi_threaded:
            result = Future()
            result.set_result(self.__do_fetch(out, *args, **kwargs))  # noqa
            return result
        else:
            return Device._workers.submit(self.__do_fetch, out, *args, **kwargs)

    def __do_fetch(self, out_, *args_, **kwargs_):
        """Helper function that awaits all futures in the keyword argument list, and then calls _fetch"""
        try:
            if len(args_) > 0 or len(kwargs_) > 0:
                logging.debug("awaiting inputs for %s (tid: %i).", self, threading.get_ident())
            awaited_args = [(arg.result() if isinstance(arg, Future) else arg) for arg in args_]
            awaited_kwargs = {key: (arg.result() if isinstance(arg, Future) else arg) for (key, arg) in kwargs_.items()}
            logging.debug("fetching data of %s ((tid: %i)).", self, threading.get_ident())
            data = self._fetch(*awaited_args, **awaited_kwargs)
            data = set_pixel_size(data, self.pixel_size)
            assert data.shape == self.data_shape
            if out_ is not None:
                out_[...] = data  # store data in the location specified during trigger
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
            self._decrease_measurements_pending()

    def __new__(cls, *args, **kwargs):
        """This method is called before __init__ to create a new instance of the class.

        We need to override this to add attributes that will be used in __setattr__
        """
        instance = super().__new__(cls)
        instance._multi_threaded = False
        return instance

    def __setattr__(self, key, value):
        """Prevents modification of public attributes and properties while the device is locked.

        For detectors, this prevents modification of the detector settings while a measurement is in progress.
        For all devices, it prevents concurrent modification in a multi-threading context.

        Private attributes can be set without locking.
        Note that this is not thread-safe and should be done with care!!
        """

        # note: the check needs to be in this order, otherwise we cannot initialize set _multi_threaded
        if not key.startswith("_") and self._multi_threaded:
            with self._lock_condition:
                while self._measurements_pending > 0:
                    self._lock_condition.wait()
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def _do_trigger(self) -> None:
        """Override this function to perform the actual hardware trigger."""
        pass

    @abstractmethod
    def _fetch(self, *args, **kwargs) -> np.ndarray:
        """Read the data from the detector

        Args:
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
    def data_shape(self) -> Tuple[int, ...]:
        """The shape of the data array that `read()` will return.

        For some detectors this property may be mutable, for example, for a camera it
        represents the height and width of the ROI, which can be changed.
        """
        return self._data_shape

    @property
    def pixel_size(self) -> Optional[Quantity]:
        """Physical dimension of one element in the returned data array.

        For cameras, this is the pixel size (in astropy length units).
        For detectors returning a time trace, this value is specified in astropy time units.
        The pixel_size is a 1-D array with an element for each dimension of the returned data.

        By default, the pixel size cannot be set.
        However, in some cases (such as when the `pixel_size` is actually a sampling interval),
        it makes sense for the child class to implement a setter.
        """
        return self._pixel_size

    @final
    def coordinates(self, dimension: int) -> Quantity:
        """Returns an array with the coordinate values along the d-th axis.

        The coordinates represent the _centers_ of the grid points. For example,
        for an array of shape ``(2,)`` the coordinates are `[0.5, 1.5] * pixel_size`
        and not `[0, 1] * pixel_size`. If `self.pixel_size is None`, a pixel size
        of 1.0 is used.

        The coordinates are returned as an array with the same number of
        dimensions as `data_shape`, with the d-th dimension holding the coordinates.
        This facilitates meshgrid-like computations, e.g.
        `cam.coordinates(0) + cam.coordinates(1)` gives a 2-dimensional array of coordinates.

        Args:
            dimension: Dimension for which to return the coordinates.
        """
        unit = u.dimensionless_unscaled if self.pixel_size is None else self.pixel_size[dimension]
        shape = np.ones_like(self.data_shape)
        shape[dimension] = self.data_shape[dimension]
        return np.arange(0.5, 0.5 + self.data_shape[dimension], 1.0).reshape(shape) * unit

    @final
    @property
    def extent(self) -> Quantity:
        """Physical size of the data array

        If a `pixel_size` is set, this function returns `data_shape * pixel_size`
        as an `astropy.units.Quantity`.
        If no `pixel_size` is set, this function uses  the
        `dimensionless_unscaled` unit.
        """
        unit = u.dimensionless_unscaled if self.pixel_size is None else self.pixel_size
        return np.array(self.data_shape) * unit


class Processor(Detector, ABC):
    """Base class for all Processors.

    Processors can be used to build data processing graphs, where each Processor takes input from one or
    more input Detectors and processes that data (e.g., cropping an image, averaging over an ROI, etc.).
    A processor, itself, is a Detector to allow chaining multiple processors together to combine functionality.

    To implement a processor, implement `_fetch`, and optionally override `data_shape`, `pixel_size`, and `__init__`.
    The `latency` and `duration` properties are computed from the latency and duration of the inputs and cannot be set.
    By default, the `pixel_size` and `data_shape` are the same as the `pixel_size` and `data_shape` of the first input.
    To override this behavior, override the `pixel_size` and `data_shape` properties.

    Args:
        multi_threaded: If True, `_fetch` is called from a worker thread. Otherwise, `_fetch` is called
                directly from `trigger`. If the device is not thread-safe, or threading provides no benefit,
                or for easy debugging, set this to False.
    """

    def __init__(self, *args, multi_threaded: bool):
        self._sources = args
        # data_shape, duration, latency and pixel_size all may change dynamically
        # when the settings of one of the source detectors is changed.
        # Therefore, we pass 'None' for all parameters, and override
        # data_shape, pixel_size, duration and latency in the properties.
        super().__init__(
            data_shape=None,
            pixel_size=None,
            duration=None,
            latency=None,
            multi_threaded=multi_threaded,
        )

    def trigger(self, *args, immediate=False, **kwargs):
        """Triggers all sources at the same time (regardless of latency), and schedules a call to `_fetch()`"""
        future_data = [
            (source.trigger(immediate=immediate) if source is not None else None) for source in self._sources
        ]
        return super().trigger(*future_data, *args, **kwargs)

    @final
    @property
    def latency(self) -> Quantity[u.ms]:
        """Returns the shortest latency for all detectors."""
        return min(
            (source.latency for source in self._sources if source is not None),
            default=0.0 * u.ms,
        )

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
        return max([duration + latency for (duration, latency) in times]) - min(
            [latency for (duration, latency) in times]
        )

    @property
    def data_shape(self):
        """This default implementation returns the data shape of the first source."""
        return self._sources[0].data_shape

    @property
    def pixel_size(self) -> Optional[Quantity]:
        """This default implementation returns the pixel size of the first source."""
        return self._sources[0].pixel_size


class PhaseSLM(ABC):
    """Base class for phase-only SLMs"""

    __slots__ = ()

    @abstractmethod
    def update(self) -> None:
        """Sends the new phase pattern to be displayed on the SLM.

        Implementations should call _start() before triggering the SLM.

        Note:
            This function *does not* wait for the image to appear on the SLM.
            To wait for the image stabilization explicitly, use 'wait()'.
            However, this should rarely be needed since all Detectors
            already wait for the image to stabilize before starting a measurement.
        """

    @abstractmethod
    def set_phases(self, values: ArrayLike, update: bool = True) -> None:
        """Sets the phase pattern on the SLM.

        Args:
            values(ArrayLike): phase pattern, in radians.
                The pattern is automatically stretched to fill the full SLM.
            update: when True, calls `update` after setting the phase pattern.
                Set to `False` to suppress the call to `update`.
                This is useful in advanced scenarios where multiple parameters of the SLM need to be changed
                before updating the displayed image.
        """
        ...
