import numpy as np
from typing import Annotated, Optional
import astropy.units as u
from astropy.units import Quantity
from enum import Enum
from concurrent.futures import Future


class NoiseType(Enum):
    UNIFORM = 1
    EXPONENTIAL = 2
    GAUSSIAN = 3


class RandomGenerator:
    """
    Generates random numbers to simulate noise images, primarily for testing device graphs.
    """

    def __init__(self, min=0, max=1000, noise_type=NoiseType.UNIFORM):
        """
        Args:
            min (int): Minimum value for random number generation.
            max (int): Maximum value for random number generation.
            noise_type (NoiseType): Type of noise to generate.
        """
        self._min = min
        self._max = max
        self._noise_type = noise_type

    def generate_into(self, buffer):
        """
        Generates random numbers into the provided buffer based on the current settings.

        Args:
            buffer (numpy.ndarray): The buffer where random numbers will be generated.
        """
        buffer[:, :] = np.random.randint(self._min, self._max, buffer.shape, dtype=np.uint16)

    @property
    def min(self) -> Annotated[int, {'min': 0, 'max': 0xFFFF}]:
        """Minimum value for random number generation, default is 0. Range: 0 to 0xFFFF."""
        return self._min

    @min.setter
    def min(self, value):
        self._min = value

    @property
    def max(self) -> Annotated[int, {'min': 0, 'max': 0xFFFF}]:
        """Maximum value for random number generation, default is 1000. Range: 0 to 0xFFFF."""
        return self._max

    @max.setter
    def max(self, value):
        self._max = value

    @property
    def noise_type(self) -> NoiseType:
        """Type of noise to generate. Currently, only uniform noise is supported."""
        return self._noise_type

    @noise_type.setter
    def noise_type(self, value):
        if not value == NoiseType.UNIFORM:
            raise ValueError("Noise types other than uniform are not supported yet.")
        self._noise_type = value


class Camera:
    """
    Simulates a camera that returns noise images, used for testing device graphs.
    Utilizes an external random number generator to produce these images.
    """

    def __init__(self, left=0, top=0, width=100, height=100, duration: Quantity[u.ms] = 100 * u.ms,
                 random_generator: Optional[RandomGenerator] = None):
        """
        Initializes the Camera with specified view dimensions, exposure time, and random generator.

        Args:
            left (int): The left coordinate of the camera's view.
            top (int): The top coordinate of the camera's view.
            width (int): The width of the camera's view.
            height (int): The height of the camera's view.
            duration (Quantity[u.ms]): The simulated exposure time.
                Default value: 100 μs
                Not really used, only serves to show the property in micromanager.
            random_generator (RandomGenerator): External random number generator to use.
        """
        if random_generator is None:
            random_generator = RandomGenerator()

        self._resized = True
        self._image = None
        self._left = left
        self._top = top
        self._width = width
        self._height = height
        self._duration = duration.to(u.ms)
        self._random_generator = random_generator

    def trigger(self):
        if self._resized:
            self._image = np.zeros(self.data_shape, dtype=np.uint16)
            self._resized = False
        self._random_generator.generate_into(self._image)
        result = Future()
        result.set_result(self._image)  # noqa
        return result

    @property
    def data_shape(self):
        return self._height, self._width

    @property
    def left(self) -> int:
        return self._top

    @left.setter
    def left(self, value: int):
        self._top = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def width(self) -> Annotated[int, {'min': 1, 'max': 1200}]:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value
        self._resized = True

    @property
    def height(self) -> Annotated[int, {'min': 1, 'max': 960}]:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value
        self._resized = True

    @property
    def duration(self) -> Quantity[u.ms]:
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value.to(u.ms)

    @property
    def random_generator(self) -> object:
        return self._random_generator

    @random_generator.setter
    def random_generator(self, value):
        self._random_generator = value


r = RandomGenerator()
devices = {'cam': Camera(random_generator=r), 'rng': r}
