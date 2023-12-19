from astropy.units import Quantity
import astropy.units as u


class GenericXYStage:
    """
    Represents a generic XY stage device with configurable step sizes and positions. This class demonstrates
    the use of astropy units for handling measurements in micrometers (um).

    Attributes:
        x (Quantity[u.um]): Current X position of the stage in micrometers.
        y (Quantity[u.um]): Current Y position of the stage in micrometers.
        step_size_x (Quantity[u.um]): Step size in the X direction in micrometers.
        step_size_y (Quantity[u.um]): Step size in the Y direction in micrometers.

    The class provides methods to control the position of the stage and adjust its step sizes.
    """
    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        """
        Initializes a new instance of the GenericXYStage class with specified step sizes.

        Args:
            step_size_x (Quantity[u.um]): The step size for movements in the X direction.
            step_size_y (Quantity[u.um]): The step size for movements in the Y direction.
        """
        super().__init__()
        self._step_size_x = step_size_x.to(u.um)
        self._step_size_y = step_size_y.to(u.um)
        self._y = 0.0 * u.um
        self._x = 0.0 * u.um

    def home(self):
        self._x = 0.0 * u.um
        self._y = 0.0 * u.um

    def busy(self):
        pass

    @property
    def x(self) -> Quantity[u.um]:
        """Current X position of the stage in micrometers."""
        return self._x

    @x.setter
    def x(self, value: Quantity[u.um]):
        former = self._x
        self._x = value.to(u.um) + former

    @property
    def y(self) -> Quantity[u.um]:
        """Current Y position of the stage in micrometers."""
        return self._y

    @y.setter
    def y(self, value: Quantity[u.um]):
        former = self._y
        self._y = value.to(u.um) + former

    @property
    def step_size_x(self) -> Quantity[u.um]:
        """Step size in the X direction in micrometers."""
        return self._step_size_x

    @step_size_x.setter
    def step_size_x(self, value: Quantity[u.um]):
        self._step_size_x = value.to(u.um)

    @property
    def step_size_y(self) -> Quantity[u.um]:
        """Step size in the Y direction in micrometers."""
        return self._step_size_y

    @step_size_y.setter
    def step_size_y(self, value: Quantity[u.um]):
        self._step_size_y = value.to(u.um)




devices = {'stage': GenericXYStage(1 * u.um, 1 * u.um)}
