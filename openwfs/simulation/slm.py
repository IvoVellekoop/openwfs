import time
from collections import deque
from typing import Sequence, Optional, Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike

from ..core import Detector, Processor, PhaseSLM, Actuator
from ..utilities import project, unitless


class PhaseToField(Processor):
    """Takes a phase as input and returns a field

    Computes `amplitude * (exp(1j * phase) + non_modulated_field_fraction)`
    """

    def __init__(
        self,
        slm_phases: Detector,
        field_amplitude: ArrayLike = 1.0,
        non_modulated_field_fraction: float = 0.0,
    ):
        """
        Args:
            slm_phases: The `Detector` that returns the phases of the slm pixels.
            field_amplitude: Field amplitude of the modulated pixels.
            non_modulated_field_fraction: Non-modulated field (e.g. a front reflection).
        """
        super().__init__(slm_phases, multi_threaded=False)
        self.modulated_field_amplitude = field_amplitude
        self.non_modulated_field = non_modulated_field_fraction

    def _fetch(self, slm_phases: np.ndarray) -> np.ndarray:  # noqa
        """
        Updates the complex field output of the SLM. The output field is the sum of the modulated field and the
        non-modulated field.
        """
        return self.modulated_field_amplitude * (np.exp(1j * slm_phases) + self.non_modulated_field)


class _SLMTiming(Detector):
    """Class to simulate the timing of an SLM.

    This class simulates latency (`update_latency`) and
    stabilization (`update_duration`) of the SLM. It does not simulate
    the refresh rate, or the conversion of gray values to phases.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        update_latency: Quantity[u.ms] = 0.0 * u.ms,
        update_duration: Quantity[u.ms] = 0.0 * u.ms,
    ):

        if len(shape) != 2:
            raise ValueError("Shape of the SLM should be 2-dimensional.")
        super().__init__(
            data_shape=shape,
            pixel_size=Quantity(2.0 / np.min(shape)),
            latency=0 * u.ms,
            duration=0 * u.ms,
            multi_threaded=False,
        )
        self.update_latency = update_latency
        self.update_duration = update_duration

        # _set_point are the voltages that the SLM hardware is currently sending to the display.
        #   The display may not have stabilized yet
        # _state are the voltages currently on the display. They will asymptotically approach _set_point
        # _state_timestamp is the time when the state was last updated
        # _queue is the queue of upcoming frames (set points and their timestamps)
        # that are sent from the PC to the SLM, but not yet displayed.
        self._set_point = np.zeros(shape, dtype=np.float32)
        self._state = np.zeros(shape, dtype=np.float32)
        self._state_timestamp = 0.0
        self._queue = deque()  # Queue to hold phase images and their timestamps

    def _fetch(self):
        return self._update(None)

    def _update(self, append):
        """Computes the currently displayed voltage image, based on the queue of set points and their timestamps.

        This function takes the frames from the queue. The ones that have a time stamp corresponding to
        the current time or earlier are merged into the current _state. The rest is kept.
        Note: if ever converting this object to multi-threading, this function should be protected by a lock.
        """
        current_time = time.time_ns() * u.ns
        while True:
            # peek the first element in the queue
            if not self._queue:
                break
            set_point, timestamp = self._queue[0]
            if timestamp > current_time:
                break

            # we found a frame that is or has been displayed
            # we step the simulation forward to the time of the frame
            self._step_to_time(timestamp)
            self._set_point = set_point
            self._queue.popleft()

        # finally, compute the current state
        if append is None:
            self._step_to_time(current_time)
        else:
            self._queue.append(append)

        return self._state

    def _step_to_time(self, timestamp: Quantity[u.ns]):
        """Step the simulation forward to a given time."""
        if self.update_duration > 0:
            a = np.exp(-(timestamp - self._state_timestamp) / self.update_duration)
            self._state = a * self._state + (1 - a) * self._set_point
        else:
            self._state = self._set_point
        self._state_timestamp = timestamp

    def send(self, phase_image):
        """Send a phase image to the SLM. This method is called by the SLM object.

        The image is not displayed directly. Instead, it is added to a queue with a timestamp
        attached, and it only takes effect after update_latency seconds have passed.

        All old images in the queue are merged with the new image, using an exponential decay.
        """
        display_time = time.time_ns() * u.ns + self.update_latency
        self._update((phase_image, display_time))


class _SLMPhaseResponse(Processor):
    def __init__(self, source: _SLMTiming, phase_response: Optional[np.ndarray] = None):
        super().__init__(source, multi_threaded=False)
        self.phase_response = phase_response

    def _fetch(self, grey_values: np.ndarray) -> np.ndarray:  # noqa
        if self.phase_response is None:
            return grey_values * (2 * np.pi / 256)
        else:
            return self.phase_response[np.rint(grey_values).astype(np.uint8)]


class SLM(PhaseSLM, Actuator):
    """
    A mock version of a phase-only spatial light modulator. Some properties are available to simulate physical
    phenomena such as imperfect phase response, and front reflections (which cause non-modulated light).
    """

    __slots__ = (
        "_hardware_fields",
        "_hardware_phases",
        "_hardware_timing",
        "_back_buffer",
        "refresh_rate",
        "_first_update_ns",
        "_lookup_table",
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        latency: Quantity[u.ms] = 0.0 * u.ms,
        duration: Quantity[u.ms] = 0.0 * u.ms,
        update_latency: Quantity[u.ms] = 0.0 * u.ms,
        update_duration: Quantity[u.ms] = 0.0 * u.ms,
        refresh_rate: Quantity[u.Hz] = 0 * u.Hz,
        field_amplitude: Union[np.ndarray, float, None] = 1.0,
        non_modulated_field_fraction: float = 0.0,
        phase_response: Optional[np.ndarray] = None,
    ):
        """

        Args:
            shape: The 2D shape of the SLM.
            field_amplitude: Field amplitude of the modulated light.
            non_modulated_field_fraction (float): fraction of the field that is not modulated,
                typically due to reflection at the front surface of the SLM.
            shape: The shape (height, width) of the SLM in pixels
            latency: The latency that the OpenWFS framework uses for synchronization.
            duration: The duration that the OpenWFS framework uses for synchronization.
            update_latency: The latency of the simulated SLM.
                Choose a value different from `latency` to simulate incorrect timing.
            update_duration: The duration of the simulated SLM.
                Choose a value different from `duration` to simulate incorrect timing.
            refresh_rate: Simulated refresh rate. Affects the timing of the `update` method,
                since this will wait until the next vertical retrace. Keep at 0 to disable this feature.
        """
        super().__init__(latency=latency, duration=duration)
        self.refresh_rate = refresh_rate
        # Simulates transferring frames to the SLM
        self._hardware_timing = _SLMTiming(shape, update_latency, update_duration)
        self._hardware_phases = _SLMPhaseResponse(
            self._hardware_timing, phase_response
        )  # Simulates reading the phase from the SLM
        self._hardware_fields = PhaseToField(
            self._hardware_phases, field_amplitude, non_modulated_field_fraction
        )  # Simulates reading the field from the SLM
        self._lookup_table = None  # index = input phase (scaled to -> [0, 255]), value = grey value
        self._first_update_ns = time.time_ns()
        self._back_buffer = np.zeros(shape, dtype=np.float32)

    def update(self):
        """Sends the current phase image to the simulated SLM hardware."""

        self._start()  # wait for detectors to finish
        if self.refresh_rate > 0:
            # wait for the vertical retrace
            time_in_frames = unitless((time.time_ns() - self._first_update_ns) * u.ns * self.refresh_rate)
            time_to_next_frame = (np.ceil(time_in_frames) - time_in_frames) / self.refresh_rate
            time.sleep(time_to_next_frame.tovalue(u.s))
            # update the start time (this is also done in the actual SLM)
            self._start()

        # Apply lookup table and compute grey values from intended phases
        # This uses the same conversion as in the shader:
        # first compute tx = phi * / (2 * pi) + 0.5 / 256
        # use the fractional part of tx to index the lookup table, where:
        # - tx =0 maps to the start of the first element
        # - tx=1-δ maps to the end of the last element
        tx = self._back_buffer * (1 / (2 * np.pi)) + (0.5 / 256)
        tx = tx - np.floor(tx)  # fractional part of tx
        if self._lookup_table is None:
            grey_values = (256 * tx).astype(np.uint8)
        else:
            lookup_index = (self._lookup_table.shape[0] * tx).astype(np.uint8)  # index into lookup table
            grey_values = self._lookup_table[lookup_index]

        self._hardware_timing.send(grey_values)

    @property
    def lookup_table(self) -> Sequence[int]:
        """Lookup table that is used to map the wrapped phase range of 0-2pi to gray values
        (represented in a range from 0 to 256). By default, this is just range(256).
        Note that the lookup table need not contain 256 elements.
        A typical scenario is to use something like `slm.lookup_table=range(142)` to map the 0-2pi range
        to only the first 142 gray values of the slm.
        """
        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, value: Sequence[int]):
        self._lookup_table = np.asarray(value)

    @property
    def phase_response(self) -> Optional[np.ndarray]:
        """Phase as a function of pixel gray value.

        This lookup table mimics the phase response of the SLM hardware.
        When it is omitted, the phase response is assumed to be linear:
        phase = 2 * pi * grey_value / 256
        """
        return self._hardware_phases.phase_response

    @phase_response.setter
    def phase_response(self, value: Sequence[float]):
        self._hardware_phases.phase_response = value

    def set_phases(self, values: ArrayLike, update=True):
        # no docstring, use documentation from base class

        # Copy the phase image to the back buffer, scaling it as necessary
        project(
            np.atleast_2d(values).astype("float32"),
            out=self._back_buffer,
            source_extent=(2.0, 2.0),
            out_extent=(2.0, 2.0),
        )
        if update:
            self.update()

    @property
    def pixels(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, these are the effective gray values that are currently displayed on the SLM.
        """
        return self._hardware_timing

    @property
    def field(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, this is the effective field that is currently on the SLM.
        """
        return self._hardware_fields

    @property
    def phases(self) -> Detector:
        """Returns an object to monitor the current state of the SLM.

        If transient behavior is simulated, this is the effective phase that is currently on the SLM.
        """
        return self._hardware_phases

    @property
    def duration(self) -> Quantity[u.ms]:
        return 0.0 * u.ms
