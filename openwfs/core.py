# core classes used throughout openwfs
import numpy as np
from astropy.units import Quantity
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Base class for all detectors, cameras and other data sources with possible dynamic behavior."""

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
    @abstractmethod
    def data_shape(self):
        """Returns a tuple corresponding to the shape of the returned data array.

        Note:
            * This value matches the `shape` property of the array returned when calling `trigger` followed by `read`.
            * The property may change, e.g., when the ROI of a camera is changed.
              In any case, the value of `data_shape`
              just before calling `trigger` will match the size of the data returned by the corresponding `read` call.
        """
        ...

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
            At the moment, the pixel_size is a scalar, so for multi-dimensional data sources the pixel_size
            must be the same in all dimensions (only square pixels are supported)
        """
        ...

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
