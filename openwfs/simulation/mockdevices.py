import time
from typing import Sequence, Optional

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from ..core import Detector, Processor, Actuator
from ..processors import CropProcessor
from ..utilities import ExtentType, get_pixel_size, set_pixel_size


class StaticSource(Detector):
    """
    Detector that returns pre-set data. Also simulates latency and measurement duration.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        pixel_size: Optional[ExtentType] = None,
        extent: Optional[ExtentType] = None,
        latency: Quantity[u.ms] = 0 * u.ms,
        duration: Quantity[u.ms] = 0 * u.ms,
        multi_threaded: bool = None,
    ):
        """
        Initializes the MockSource
        TODO: factor out the latency and duration into a separate class?
        Args:
            data (np.ndarray): The pre-set data to be returned by the mock source.
            pixel_size (Quantity, optional): The size of each pixel in the data.
                If not specified, the pixel size is calculated from the extent and the data shape.
                If neither pixel size nor extent are specified, the pixel size from `data` is used, if available.
                Otherwise, pixel_size is set to None.
            extent (Optional[ExtentType]): The physical extent of the data array.
                Only used when the pixel size is not specified explicitly.
        """
        if pixel_size is None:
            if extent is not None:
                pixel_size = Quantity(extent) / data.shape
            else:
                pixel_size = get_pixel_size(data)
        else:
            data = set_pixel_size(data, pixel_size)  # make sure the data array holds the pixel size

        if pixel_size is not None and (np.isscalar(pixel_size) or pixel_size.size == 1) and data.ndim > 1:
            pixel_size = pixel_size.repeat(data.ndim)

        if multi_threaded is None:
            multi_threaded = latency > 0 * u.ms or duration > 0 * u.ms

        self._data = data
        super().__init__(
            data_shape=data.shape,
            pixel_size=pixel_size,
            latency=latency,
            duration=duration,
            multi_threaded=multi_threaded,
        )

    def _fetch(self) -> np.ndarray:  # noqa
        total_time_s = self.latency.to_value(u.s) + self.duration.to_value(u.s)
        if total_time_s > 0.0:
            time.sleep(total_time_s)
        return self._data

    @property
    def data(self):
        """
        The pre-set data to be returned by the mock source.

        Note:
            The data is not copied.
            When setting the `data` property, the `data_shape` attribute is updated accordingly.
            If the `data` property is set with an array that has the `pixel_size` metadata set,
            the `pixel_size` attribute is also updated accordingly.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._pixel_size = get_pixel_size(value)
        self._data_shape = value.shape


class NoiseSource(Detector):
    def __init__(
        self,
        noise_type: str,
        *,
        data_shape: tuple[int, ...],
        pixel_size: Quantity,
        multi_threaded=True,
        generator=None,
        **kwargs,
    ):
        self._noise_type = noise_type
        self._noise_arguments = kwargs
        self._rng = generator if generator is not None else np.random.default_rng()
        super().__init__(
            data_shape=data_shape,
            pixel_size=pixel_size,
            latency=0 * u.ms,
            duration=0 * u.ms,
            multi_threaded=multi_threaded,
        )

    def _do_trigger(self) -> None:
        pass  # no hardware triggering is needed for this mock device

    def _fetch(self) -> np.ndarray:  # noqa
        if self._noise_type == "uniform":
            return self._rng.uniform(**self._noise_arguments, size=self.data_shape)
        elif self._noise_type == "gaussian":
            return self._rng.normal(**self._noise_arguments, size=self.data_shape)
        else:
            raise ValueError(f"Unknown noise type: {self._noise_type}")

    @Detector.data_shape.setter
    def data_shape(self, value):
        self._data_shape = tuple(value)


class ADCProcessor(Processor):
    """Mimics an analog-digital converter.

    At the moment, only positive input and output values are supported.
    """

    def __init__(
        self,
        source: Detector,
        analog_max: Optional[float],
        digital_max: int = 0xFFFF,
        shot_noise: bool = False,
        gaussian_noise_std: float = 0.0,
        multi_threaded: bool = True,
        generator=None,
    ):
        """
        Initializes the ADCProcessor class, which mimics an analog-digital converter.

        Args:
            source: The source detector providing analog data.
            analog_max: The maximum analog value that can be handled by the ADC.
                If set to None, each measurement will be automatically scaled so that the maximum
                value in the data set returned by `source` is converted to `digital_max`.
                Note that this means that the values from two different measurements cannot be compared quantitatively.

            digital_max:
                The maximum digital value that the ADC can output, default is unsigned 16-bit maximum.

            shot_noise:
                Flag to determine if Poisson noise should be applied instead of rounding.
                Useful for realistically simulating detectors.

            gaussian_noise_std:
                If >0, add gaussian noise with std of this value to the data.
        """
        super().__init__(source, multi_threaded=multi_threaded)
        self._analog_max = None
        self._digital_max = None
        self._shot_noise = False
        self._gaussian_noise_std = 0.0
        self._rng = generator if generator is not None else np.random.default_rng()
        self.gaussian_noise_std = gaussian_noise_std
        self.shot_noise = shot_noise
        self.analog_max = analog_max  # check value
        self.digital_max = digital_max  # check value

    def _fetch(self, data) -> np.ndarray:  # noqa
        """Clips the data to the range of the ADC, and digitizes the values."""

        if self.analog_max is None:  # auto scaling
            max_value = np.max(data)
            if max_value > 0.0:
                data = data * (self.digital_max / max_value)  # auto-scale to maximum value
        else:
            data = data * (self.digital_max / self.analog_max)

        if self._shot_noise:
            data = self._rng.poisson(data)

        if self._gaussian_noise_std > 0.0:
            data = data + self._rng.normal(scale=self._gaussian_noise_std, size=data.shape)

        return np.clip(np.rint(data), 0, self.digital_max).astype("uint16")

    @property
    def analog_max(self) -> Optional[float]:
        """Maximum value that the ADC can handle as input

        This value and all higher values are converted to `digital_max`.
        When set to 0.0, the input signal is scaled automatically so that the maximum corresponds to
        `digital_max`
        """
        return self._analog_max

    @analog_max.setter
    def analog_max(self, value: Optional[float]):
        if value is None:
            self._analog_max = None
            return
        if value < 0.0:
            raise ValueError("analog_max cannot be negative")
        self._analog_max = float(value)

    @property
    def digital_max(self) -> int:
        """Maximum value that the ADC can output.

        Default value is 0xFFFF (16 bits)
        """
        return self._digital_max

    @property
    def conversion_factor(self) -> Optional[float]:
        """Conversion factor between analog and digital values.
        If analog_max is set to None, each frame is auto-scaled, and this function returns None.
        """
        return self.digital_max / self.analog_max if self.analog_max is not None else None

    @digital_max.setter
    def digital_max(self, value):
        if value < 0 or value > 0xFFFF:
            raise ValueError("digital_max must be between 0 and 0xFFFF")
        self._digital_max = int(value)

    @property
    def shot_noise(self) -> bool:
        """when True, apply Poisson noise to the data instead of rounding"""
        return self._shot_noise

    @shot_noise.setter
    def shot_noise(self, value: bool):
        self._shot_noise = bool(value)

    @property
    def gaussian_noise_std(self) -> float:
        return self._gaussian_noise_std

    @gaussian_noise_std.setter
    def gaussian_noise_std(self, value: float):
        self._gaussian_noise_std = float(value)


class Camera(ADCProcessor):
    """Wraps any 2-D image source as a camera.

    To implement the camera interface, in addition to the Detector interface,
    we must implement left,right,top, and bottom.
    In addition, the data should be returned as uint16.
    Conversion to uint16 is implemented in the ADCProcessor base class.
    """

    def __init__(
        self,
        source: Detector,
        shape: Optional[Sequence[int]] = None,
        pos: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """
        Args:
            source (Detector): The source detector to be wrapped.
            shape (Optional[Sequence[int]]): The shape of the image data to be captured.
            pos (Optional[Sequence[int]]): The position on the source from where the image is captured.
            **kwargs: Additional keyword arguments to be passed to the Detector base class.

            TODO: move left-right-top-bottom to CropProcessor.
            Expose the properties of CropProcessor as properties of MockCamera automatically by copying from __dict__?
        """
        self._crop = CropProcessor(source, shape=shape, pos=pos)
        super().__init__(source=self._crop, **kwargs)

    @property
    def left(self) -> int:
        """left (int): Horizontal start position of the ROI."""
        return self._crop.pos[1]

    @left.setter
    def left(self, value: int):
        self._crop.pos = (self._crop.pos[0], value)

    @property
    def right(self) -> int:
        """right (int): Horizontal end position of the ROI."""
        return self.left + self.width

    @property
    def top(self) -> int:
        """top (int): Vertical start position of the ROI."""
        return self._crop.pos[0]

    @top.setter
    def top(self, value: int):
        self._crop.pos = (value, self._crop.pos[1])

    @property
    def bottom(self) -> int:
        """bottom (int): Vertical end position of the ROI."""
        return self.top + self.height

    @property
    def height(self) -> int:
        """Height of the ROI."""
        return self._crop.data_shape[0]

    @height.setter
    def height(self, value: int):
        self._crop.data_shape = (value, self._crop.data_shape[1])

    @property
    def width(self) -> int:
        """Width of the ROI."""
        return self._crop.data_shape[1]

    @width.setter
    def width(self, value: int):
        self._crop.data_shape = (self._crop.data_shape[0], value)

    @property
    def data_shape(self):
        return self._crop.data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._crop.data_shape = value

    @property
    def exposure(self) -> Quantity[u.ms]:
        return self.duration


class XYStage(Actuator):
    """
    Mimics an XY stage actuator
    """

    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        """

        Args:
            step_size_x (Quantity[u.um]): The step size in the x-direction.
            step_size_y (Quantity[u.um]): The step size in the y-direction.
        """
        super().__init__(duration=0 * u.ms, latency=0 * u.ms)
        self._step_size_x = step_size_x.to(u.um)
        self._step_size_y = step_size_y.to(u.um)
        self._y = 0.0 * u.um
        self._x = 0.0 * u.um

    @property
    def step_size_x(self) -> Quantity[u.um]:
        return self._step_size_x

    @property
    def step_size_y(self) -> Quantity[u.um]:
        return self._step_size_y

    @property
    def x(self) -> Quantity[u.um]:
        return self._x

    @x.setter
    def x(self, value: Quantity[u.um]):
        self._x = self.step_size_x * np.round(value.to(u.um) / self.step_size_x)

    @property
    def y(self) -> Quantity[u.um]:
        return self._y

    @y.setter
    def y(self, value: Quantity[u.um]):
        self._y = self.step_size_y * np.round(value.to(u.um) / self.step_size_y)

    def home(self):
        self._x = 0.0 * u.um
        self._y = 0.0 * u.um


class Shutter(Processor):
    """
    A mock version of a shutter.
    When open, passes through the input field
    When closed, passes through the input field * 0.0
    """

    def __init__(self, source: Detector):
        super().__init__(source, multi_threaded=False)
        self._open = True

    @property
    def open(self) -> bool:
        return self._open

    @open.setter
    def open(self, value: bool):
        self._open = value

    def _fetch(self, source: np.ndarray) -> np.ndarray:  # noqa
        return source if self._open else 0.0 * source


class GaussianNoise(Processor):
    """Adds gaussian noise of a specified standard deviation to the signal
    Args:
        source (Detector): The source detector object to process the data from.
        std (float): The standard deviation of the gaussian noise.
        multi_threaded: Whether to perform processing in a worker thread.
    """

    def __init__(self, source: Detector, std: float, multi_threaded: bool = False):
        super().__init__(source, multi_threaded=multi_threaded)
        self._std = std

    @property
    def std(self) -> float:
        return self._std

    @std.setter
    def std(self, value: float):
        if value < 0.0:
            raise ValueError("Standard deviation must be non-negative")
        self._std = float(value)

    def _fetch(self, data: np.ndarray) -> np.ndarray:  # noqa
        """
        Args:
            data (ndarray): source data

        Returns: the out array containing the image with added noise.

        """
        return data + np.random.normal(0.0, self.std, data.shape)
