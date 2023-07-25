import numpy as np
from typing import Protocol


class Detector(Protocol):
    """Minimal implementation of a detector object. Any detector must implement these members and attributes"""

    data_shape: tuple[int]
    """shape of the data returned by this detector (from numpy.shape). Read only."""

    measurement_time: float
    """In seconds. Read only. Used for synchronization, and should include the measurement time of any child 
    detectors (see Processor)."""

    def trigger(self) -> None:
        """Triggers the detector to take a new measurement. Typically, does not wait until the measurement is
        finished. A detector may have a hardware trigger, in which case calls to trigger() may be ignored."""
        pass

    def read(self) -> np.ndarray:
        """Returns the measured data, in the order that the triggers were given. This function blocks until the data
        is available and raises a TimeoutError if it takes too long to obtain the data (typically because the detector
        was not triggered)."""
        pass


class MockDetector:  # implements Detector
    """Fake detector that returns random values between 0.0 and 1.0."""

    def __init__(self, data_shape):
        self.data_shape = data_shape  # shape of the data returned by this detector (from numpy.shape). Read only.
        self.measurement_time = 0.0  # time in seconds that a measurement takes. Used for synchronization. Read only.
        self._buffer = []  # only for demonstration purpose, not needed

    def trigger(self):
        """Triggers the detector to take a new measurement. Typically, does not wait until the measurement is
        finished. A detector may have a hardware trigger, in which case calls to trigger() may be ignored."""
        self._buffer.append(np.random.rand(self.data_shape))

    def read(self):
        """Returns the measured data, in the order that the triggers were given. This function blocks until the data
        is available and raises a TimeoutError if it takes too long to obtain the data (typically because the detector
        was not triggered)."""
        if self._buffer.count == 0:
            raise TimeoutError()
        return self._buffer.pop(0)


class Controller:
    def __init__(self, detector: Detector, slm):
        self.M = None
        """ Number of elements in each measurement. Read only. Equal to np.prod(source.data_shape).
            Updated when 'reserve' is called"""

        self.N = None
        """ Total number of measurements in the buffer. Read only. Set by 'reserve' function. """

        self.data_shape = detector.data_shape
        """ Shape of the original data, before flattening into column of length M."""

        self.slm = slm
        """Spatial light modulator object. An algorithm sets or changes slm.phases, and then calls measure() to 
        schedule a measurement. Note that there is no need to call slm.update manually."""

        self._measurements_pending = 0  # number of measurements that were triggered but not yet read
        self._measurements = None  # array storing all measurements. shape e.g. (Nx, Ny, M)
        self._measurements_flat = None  # reshaped N x M view of the same array
        self._source = detector  # detector to take the feedback from.
        self._n = None  # current measurement number, must be < N

    def __del__(self):
        # clear remaining measurements. todo: implement aborting measurements?
        while self._measurements_pending > 0:
            self._await_data()

    def reserve(self, shape):
        """ Reserve space for storing measurements. Must be called before the first call to 'measure'. """
        self._n = 0
        self.N = np.prod(shape)
        self.M = np.prod(self._source.data_shape)
        self.data_shape = self._source.data_shape
        self._measurements = np.zeros((*shape, self.M), dtype="float32")
        self._measurements_flat = np.reshape(self._measurements, (self.N, self.M))

    def measure(self):
        """ Schedule a measurement. A measurements corresponds to updating the SLM, waiting for the image to
        stabilize, triggering the detector and reading the data from the detector. There is no guarantee as to when
        this measurement is performed (measurements may even be performed out of order or batched together in some
        implementations). However, the data is guaranteed to end up in the 'measurements' array in the correct order."""

        # Update the SLM. If the SLM was reserved, first wait until the measurement_time - idle_time has passed before
        # flipping the buffers. Don't wait for the SLM to stabilize
        self.slm.update(wait=False)

        # we can use this time to process a previous measurement (if any)
        self._await_data()

        self.slm.wait()  # wait for the image on the SLM to stabilize
        self._source.trigger()  # trigger the camera
        self._measurements_pending += 1

        measurement_time = self._source.measurement_time
        if measurement_time is not None:
            # default fast wavefront shaping, continue processing (and even start next frame) during measurement
            self.slm.reserve(measurement_time)
        else:
            # measurement time not known, wait till end of measurement
            self._await_data()

    def _await_data(self):
        """ Reads data from the detector (which includes processing it). If there are no outstanding measurements,
        this function does nothing. Note that other implementations of feedback may choose to batch-read data from
        the device and not even implement this function"""
        if self._measurements_pending > 0:
            data = np.array(self._source.read(), dtype="float32", copy=False)
            self._measurements_pending -= 1
            self._measurements_flat[self._n, :] = data.flat
            self._n += 1

    @property
    def measurements(self, partial=False):
        """ Called after performing a sequence of measurements to obtain the measurement data. Typically,
        you only call this function after performing all measurements that your reserved space for.
        If you want to peek before finishing all measurements, set 'partial'=True"""
        while self._measurements_pending > 0:
            self._await_data()
        if not partial and self._n != self.N:
            raise Exception(f"Measurement sequence not completed yet, only performed {self._n} out of {self.N} "
                            f"measurements.")
        return self._measurements


    def compute_transmission(self, phase_steps):
        """To do: calculate SnR"""
        t = np.tensordot(self.measurements, np.exp(-1j * np.arange(phase_steps) / phase_steps * 2 * np.pi),
                         (len(np.shape(self.measurements))-2, [0]))  # phase steps should be in second to last dimension
        return t
