from astropy.units import Quantity
import astropy.units as u


class GenericZStage:
    """
    Represents a generic Z stage device, primarily used for vertical (Z-axis) movements. The class
    demonstrates the use of astropy units for handling measurements in micrometers (μm).

    The class allows setting and retrieving the position and step size of the stage, enabling precise control
    over vertical movements.
    """

    def __init__(self, step_size: Quantity[u.um]):
        """
        Initializes a new instance of the GenericZStage class with a specified step size.

        Args:
            step_size (Quantity[u.um]): The step size for vertical movements.
        """
        super().__init__()
        self._step_size = step_size.to(u.um)

        self._position = 0.0 * u.um

    def home(self):
        self._position = 0.0 * u.um

    def busy(self):
        pass

    @property
    def position(self) -> Quantity[u.um]:
        """Current position of the stage in micrometers."""
        return self._position

    @position.setter
    def position(self, value: Quantity[u.um]):
        former = self._position
        self._position = value.to(u.um) + former

    @property
    def step_size(self) -> Quantity[u.um]:
        """Step size for movements in micrometers."""
        return self._step_size

    @step_size.setter
    def step_size(self, value: Quantity[u.um]):
        self._step_size = value.to(u.um)


devices = {'stage': GenericZStage(1 * u.um)}

