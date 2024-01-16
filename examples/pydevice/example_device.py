from typing import Annotated
from enum import Enum
import astropy.units as u


class SomeOptions(Enum):
    Orange = 1
    Red = 2
    Blue = 3


class GenericDevice:
    """
    Represents a generic device with various configurable options. This device serves as a demonstration
    of different data types and their handling.

    The class allows setting and retrieving these attributes, demonstrating the use of properties
    and type annotations in Python.
    """

    def __init__(self, options, floating_point, distance, boolean, integer, command):
        """
        Initializes a new instance of the GenericDevice class with specified configurations.

        Args:
            options (SomeOptions): The initial setting for the device options.
            floating_point (float): The initial floating-point value.
            distance (u.Quantity[u.mm]): The initial distance measurement.
            boolean (bool): The initial boolean state.
            integer (int): The initial integer value.
            command (str): The initial command for the device.
        """
        self._options = options
        self._floating_point = floating_point
        self._distance = distance
        self._boolean = boolean
        self._integer = integer
        self._command = command

    @property
    def options(self) -> SomeOptions:
        """A property that takes values from the `SomeOptions` enumeration."""
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    @property
    def floating_point(self) -> float:
        """A floating-point value representing some parameter of the device."""
        return self._floating_point

    @floating_point.setter
    def floating_point(self, value):
        self._floating_point = value

    @property
    def distance(self) -> u.Quantity[u.mm]:
        """A distance measurement with astropy units, in millimeters."""
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value.to(u.mm)

    @property
    def boolean(self) -> bool:
        """A boolean value representing a binary state or choice for the device."""
        return self._boolean

    @boolean.setter
    # setting value:bool forces the users to input the correct type. Optional.
    def boolean(self, value: bool):
        self._boolean = value

    @property
    # setting a range, also sets this range in MicroManager, also optional.
    def integer(self) -> Annotated[int, {'min': 0, 'max': 42}]:
        """An integer value, constrained within a specified range (0 to 42)."""
        return self._integer

    @integer.setter
    def integer(self, value):
        self._integer = value

    @property
    def command(self) -> str:
        """command (str): The initial command for the device."""
        return self._command

    @command.setter
    def command(self, value):
        self._command = str(value)


device = GenericDevice(SomeOptions.Blue, 23.7, 0.039 * u.m, True, 4, 'Something')
devices = {'some_device': device}
