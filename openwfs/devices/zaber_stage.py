import threading
import weakref

import astropy.units as u
import numpy as np
import serial.tools.list_ports
from astropy.units import Quantity
from zaber_motion import Units, Library, ascii, binary

from ..core import Actuator


class _SerialPortConnection:
    """
    Manages a single serial port connection shared among multiple users.
    Uses weak references to automatically close the connection when no users remain.
    """

    def __init__(self, port: str, protocol: str):
        self.protocol = protocol.lower()
        if self.protocol == "ascii":
            self.connection = ascii.Connection.open_serial_port(port)
        elif self.protocol == "binary":
            self.connection = binary.Connection.open_serial_port(port)
        else:
            raise ValueError("protocol must be 'ascii' or 'binary'")

    def __del__(self):
        self.connection.close()

    _ports = weakref.WeakValueDictionary()
    _lock = threading.RLock()

    @staticmethod
    def open(port: str, protocol: str | None = None) -> "_SerialPortConnection":
        """Opens a connection to a serial port.

        Establishes or retrieves an existing serial port connection, ensuring that only one connection
        is allowed per port with the same protocol. If a connection for the given port exists but uses
        a different protocol, an exception is raised.

        Args:
            port: The identifier of the serial port, e.g. "COM3"
            protocol ("ascii" | "binary" | None):
                The protocol associated with the connection.
                When 'None' and the port was already open, use the protocol with which it was opened earlier.
                If no port was open, use "ascii".

        Returns:
            _SerialPortConnection: A new or existing serial port connection for the given port.

        Raises:
            RuntimeError: If the specified port is already open with a different protocol.
        """
        with _SerialPortConnection._lock:
            connection = _SerialPortConnection._ports.get(port)
            if connection is not None:
                if protocol is not None and connection.protocol != protocol:
                    raise RuntimeError(f"Port {port} is already open with a different connection class.")
            else:
                connection = _SerialPortConnection(port, protocol or "ascii")
                _SerialPortConnection._ports[port] = connection

            return connection


class _ZaberConnection:
    """
    Base class for Zaber devices connected via serial ports, handles protocol selection and connection setup.

    A _ZaberConnection object holds a connection to a serial port,
    as well as a device number to identify which device on the port to control
    (since there may be multiple devices on the same port).

    Args:
        port: COM port string, e.g. "COM3"
        device_number: index of the device on the port (0-based)
        protocol: "ascii" or "binary"

    Example:
        conn = _ZaberConnection(port="COM3", device_number=0, protocol="ascii
    """

    def __init__(self, port: str, device_number: int = 0, protocol: str = "ascii"):
        # open an ascii or binary connection to the port
        self.connection = _SerialPortConnection.open(port, protocol)
        self.device_number = device_number
        try:
            self.device = self.connection.connection.detect_devices()[device_number]
        except IndexError:
            raise RuntimeError(f"No Zaber devices found on port {port} using {protocol} protocol.")

    @staticmethod
    def list_all_devices():
        """
        List all COM ports and devices connected to them using both ASCII and Binary protocols.
        Works even if some ports are already open by objects.

        Returns: dict { "COMx": [{"protocol": "ascii"|"binary", "devices": [...] }], ... }
        """
        all_devices = {}
        for port in serial.tools.list_ports.comports():
            connection = _SerialPortConnection.open(port)
            print(f"Port: {port}")
            try:
                devices = connection.connection.detect_devices()
                all_devices[port] = {"protocol": connection.protocol, "devices": devices}
                print(f"  Protocol: {connection.protocol}")
                for i, d in enumerate(devices):
                    print(f"    Device {i}: {d}")
            except Exception as e:
                print(f"[WARN] Could not query existing connection {port.device}: {e}")

        return all_devices


class ZaberXYStage(Actuator):
    """
    Wrapper for a pair of Zaber linear stages connected via serial ports, controlling X and Y axes.

    todo: if we add more stages, it makes sense to make a general XYStage class that combines two linear stages into one.
    todo: Currently both axis move after each other, need to implement parallel movement.

    Args:
        port_x: COM port string for X axis, e.g. "COM3"
        port_y: COM port string for Y axis, e.g. "COM4". If None, uses port_x.
        device_number_x: index of the device on the X axis port (0-based)
        device_number_y: index of the device on the Y axis port (0-based). If None and port_y==port_x, assumes second device (1), else 0.
        protocol: "ascii" or "binary"

    Example:
        stage = ZaberXYStage(port_x="COM3", port_y="COM4", device_number_x=0, device_number_y=0, protocol="ascii")
        stage.x = 5000 * u.um  # move X to 5000 micrometers
        stage.y = 2 * u.mm  # move Y to 2 millimeters
        print(stage.x, stage.y)  # get current positions
        stage.home()  # home both axes
    """

    def __init__(
        self,
        port_x: str,
        port_y: str | None = None,
        device_number_x: int = 0,
        device_number_y: int | None = None,
        protocol: str = "ascii",
    ):
        # Initialize base class
        super().__init__(self, duration=np.inf * u.ms, latency=0 * u.ms)

        # If only one port is given, use it for both axes
        if port_y is None:
            port_y = port_x

        # If only one port is given and device_number_y is not specified, assume second device on same port and set device_number_y=1,
        # otherwise set device_number_y=0
        if device_number_y is None:
            device_number_y = 1 if port_y == port_x else 0

        self.port_x = port_x  # save ports
        self.port_y = port_y

        # Create stage objects
        self.stage_x = ZaberLinearStage(port_x, device_number=device_number_x, protocol=protocol)
        self.stage_y = ZaberLinearStage(port_y, device_number=device_number_y, protocol=protocol)

    @property
    def x(self) -> Quantity[u.um]:
        return self.stage_x.x

    @x.setter
    def x(self, value: Quantity[u.um]):
        self.stage_x.x = value

    @property
    def y(self) -> Quantity[u.um]:
        return self.stage_y.x

    @y.setter
    def y(self, value: Quantity[u.um]):
        self.stage_y.x = value

    def home(self):
        self.stage_x.home()
        self.stage_y.home()
        return self.x, self.y

    def x_home(self):
        self.stage_x.home()
        return self.x

    def y_home(self):
        self.stage_y.home()
        return self.y

    @staticmethod
    def list_all_devices():
        return _ZaberConnection.list_all_devices()


class ZaberLinearStage(Actuator):
    """
    Wrapper for a single Zaber linear stage connected via serial port.

    Args:
        port: COM port string, e.g. "COM3"
        device_number: index of the device on the port (0-based)
        protocol: "ascii" or "binary"

    Example:
        stage = ZaberLinearStage(port="COM3", device_number=0, protocol="ascii")
        stage.x = 5000 * u.um  # move to 5000 micrometers
        print(stage.x)  # get current position
        stage.home()  # home the stage
    """

    def __init__(self, port: str, device_number: int = 0, protocol: str = "ascii"):
        # Initialize base class
        Actuator.__init__(self, duration=0 * u.ms, latency=0 * u.ms)
        self.serial_port = _ZaberConnection(port, device_number=device_number, protocol=protocol)
        self.stage = self.serial_port.device  # the Zaber device object
        self.device_number = device_number  # save device number

    @property
    def x(self) -> Quantity[u.um]:
        return self.stage.get_position(unit=Units.LENGTH_MICROMETRES) * u.um

    @x.setter
    def x(self, value: Quantity[u.um]):
        self.stage.move_absolute(value.to(u.um).value, Units.LENGTH_MICROMETRES)

    def home(self):
        self.stage.home()
        return self.x

    @staticmethod
    def list_all_devices():
        return _ZaberConnection.list_all_devices()
