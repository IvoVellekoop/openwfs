import threading
import weakref

import astropy.units as u
import serial.tools.list_ports
from astropy.units import Quantity
from zaber_motion import Units, Library
from zaber_motion import ascii, binary

from ..core import Actuator


class _SerialPortConnection:
    """
    Manages a single serial port connection shared among multiple users.
    Uses weak references to automatically close the connection when no users remain.
    Ensures thread-safe access to the shared connections.
    """

    def __init__(self, connection_class, port):
        self.connection_class = connection_class
        self.connections = connection_class.open_serial_port(port)

    def __del__(self):
        self.connections.close()

    _ports = weakref.WeakValueDictionary()
    _lock = threading.RLock()

    @staticmethod
    def open(connection_class, port: str) -> "_SerialPortConnection":
        with _SerialPortConnection._lock:
            conn = _SerialPortConnection._ports.get(port)
            if conn is not None:
                if conn.connection_class != connection_class:
                    raise RuntimeError(f"Port {port} is already open with a different connection class.")
            else:
                conn = _SerialPortConnection(connection_class, port)
                _SerialPortConnection._ports[port] = conn

            return conn


class _ZaberConnection:
    """
    Base class for Zaber devices connected via serial ports, handles protocol selection and connection setup.
    Args:
        port: COM port string, e.g. "COM3"
        device_number: index of the device on the port (0-based)
        protocol: "ascii" or "binary"

    Example:
        conn = _ZaberConnection(port="COM3", device_number=0, protocol="ascii
    """

    def __init__(self, port: str, device_number: int = 0, protocol: str = "ascii"):
        self.protocol = protocol.lower()
        if self.protocol == "ascii":
            ConnectionClass = ascii.Connection
        elif self.protocol == "binary":
            ConnectionClass = binary.Connection
        else:
            raise ValueError("protocol must be 'ascii' or 'binary'")

        self.connection = _SerialPortConnection.open(ConnectionClass, port)
        self.device_number = device_number

        # Detect devices (slow operations outside the lock)
        devices = self.connection.connections.detect_devices()
        if not devices:
            raise RuntimeError(f"No Zaber devices found on port {port} using {protocol} protocol.")
        self.device = devices[device_number]

    @staticmethod
    def list_all_devices():
        """
        List all COM ports and devices connected to them using both ASCII and Binary protocols.
        Works even if some ports are already open by objects.

        Returns: dict { "COMx": [{"protocol": "ascii"|"binary", "devices": [...] }], ... }
        """
        devices_found = {}
        ports = serial.tools.list_ports.comports()

        for port in ports:
            port_info = []

            # Check if port is already open
            with _SerialPortConnection._lock:
                conn = _SerialPortConnection._ports.get(port.device)

            if conn is not None:
                try:
                    devices = conn.connections.detect_devices()
                    if devices:
                        proto_name = "ascii" if isinstance(conn.connection_class, ascii.Connection) else "binary"
                        port_info.append({"protocol": proto_name, "devices": devices})
                except Exception as e:
                    print(f"[WARN] Could not query existing connection {port.device}: {e}")
            else:
                # Probe with temporary connections
                for proto_name, ConnectionClass in [("ascii", ascii.Connection), ("binary", binary.Connection)]:
                    try:
                        tmp = ConnectionClass.open_serial_port(port.device)
                        try:
                            devices = tmp.detect_devices()
                            if devices:
                                port_info.append({"protocol": proto_name, "devices": devices})
                        finally:
                            try:
                                tmp.close()
                            except Exception:
                                pass
                    except Exception:
                        continue

            if port_info:
                devices_found[port.device] = port_info

        return devices_found


class ZaberXYStage(Actuator):
    """
    Wrapper for a pair of Zaber linear stages connected via serial ports, controlling X and Y axes.

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

    ToDo: Currently both axis move after each other, need to implement parallel movement.
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
        Actuator.__init__(self, duration=0 * u.ms, latency=0 * u.ms)

        # If only one port is given, use it for both axes
        if port_y is None:
            port_y = port_x

        # If only one port is given and device_number_y is not specified, assume second device on same port and set device_number_y=1,
        # otherwise set device_number_y=0
        if device_number_y is None and port_y == port_x:
            device_number_y = 1  # second device on same port
        elif device_number_y is None:
            device_number_y = 0  # second device on different port

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
        return self.x

    @property
    def y(self) -> Quantity[u.um]:
        return self.stage_y.x

    @y.setter
    def y(self, value: Quantity[u.um]):
        self.stage_y.x = value
        return self.y

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
        return self.x

    def home(self):
        self.stage.home()
        return self.x

    def list_all_devices():
        return _ZaberConnection.list_all_devices()
