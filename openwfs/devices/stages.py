import astropy.units as u
from astropy.units import Quantity
from zaber_motion import Units, Library
from zaber_motion import ascii, binary
import serial.tools.list_ports
from ..core import Actuator
import weakref


class SerialPortBase:
    """
    Base class for devices connected via serial ports, handles protocol selection,
    connection setup, device detection, and shared connections.
    """

    # Class-level caches for connections (keeping track of all connected devices, only close port if all devices connected that port are closed)

    _connections = weakref.WeakValueDictionary()  # port -> connection (weakref to allow auto-cleanup)

    def __init__(self, protocol: str = "ascii"):
        """
        Args:
            protocol (str): "ascii" or "binary" (default "ascii")
        """
        self.protocol = protocol.lower()
        if self.protocol == "ascii":
            self.ConnectionClass = ascii.Connection
        elif self.protocol == "binary":
            self.ConnectionClass = binary.Connection
        else:
            raise ValueError("protocol must be 'ascii' or 'binary'")

        self._closed = False  # prevents double-close per object
        self.port = None  # to be set by subclasses

    def open_connection(self, port: str):
        """
        Open a serial port connection with the chosen protocol, or reuse an existing one.
        Returns the connection object.
        """

        conn = SerialPortBase._connections.get(port)
        if conn is not None:
            # reuse existing connection
            self._connection = conn
            self._port = port
            return conn

        # Open new connection
        conn = self.ConnectionClass.open_serial_port(port)
        SerialPortBase._connections[port] = conn

        # add to weak dict (auto-removed when no strong refs left)
        SerialPortBase._connections[port] = conn
        self._connection = conn
        self._port = port

        # register finalizer: close when GC runs out of refs
        weakref.finalize(conn, conn.close)
        return conn

    def close(self):
        """Explicitly close this object’s connection and remove from cache."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                print(f"[WARN] Error closing {self._port}: {e}")
            finally:
                # Remove from the WeakValueDictionary immediately
                if self._port in SerialPortBase._connections:
                    try:
                        del SerialPortBase._connections[self._port]
                    except KeyError:
                        pass
                self._connection = None
            self._port = None

    @staticmethod
    def list_all_devices():
        """
        List all COM ports and devices connected to them using both ASCII and Binary protocols.
        Works even if some ports are already open by objects.

        Returns a dict: { "COMx": [{"protocol": "ascii", "devices": [...]}, {"protocol": "binary", "devices": [...]}], ... }
        """
        devices_found = {}
        ports = serial.tools.list_ports.comports()

        for port in ports:
            port_info = []

            # First check if we already have this port open
            if port.device in SerialPortBase._connections:
                conn = SerialPortBase._connections[port.device]
                try:
                    devices = conn.detect_devices()
                    if devices:
                        # we don't know protocol directly, so infer from conn class
                        proto_name = "ascii" if isinstance(conn, ascii.Connection) else "binary"
                        port_info.append({"protocol": proto_name, "devices": devices})
                except Exception as e:
                    print(f"[WARN] Could not query existing connection {port.device}: {e}")

            else:
                # Otherwise, try opening fresh with both protocols
                for proto_name, ConnectionClass in [("ascii", ascii.Connection), ("binary", binary.Connection)]:
                    try:
                        with ConnectionClass.open_serial_port(port.device) as conn:
                            devices = conn.detect_devices()
                            if devices:
                                port_info.append({"protocol": proto_name, "devices": devices})
                    except Exception:
                        continue

            if port_info:
                devices_found[port.device] = port_info

        # Print nicely
        for port, info_list in devices_found.items():
            print(f"Port: {port}")
            for info in info_list:
                proto = info["protocol"]
                devs = info["devices"]
                print(f"  Protocol: {proto}")
                for i, d in enumerate(devs):
                    print(f"    Device {i}: {d}")

        return devices_found

    @classmethod
    def _print_counters(cls):
        """Print all active connection counters."""
        print("\n=== Connection counters ===")
        if not cls._connections:
            print("No open connections.")
            return

        for port, conn in cls._connections.items():
            count = 31415  # cls._ref_counts.get(port, 0)
            print(f"Port: {port}, Ref count: {count}, Connection: {conn}")


class ZaberXYStage(Actuator, SerialPortBase):
    """
    Wraps two Zaber stages (x and y axes) so they look like an XYStage.
    Handles connection setup given one or two COM ports.

    Args:
        port_x (str): COM port name for the x-axis stage (e.g. "COM3").
        port_y (str | None): COM port name for the y-axis stage (e.g. "COM4").
                             If None, assume both axes are connected on port_x.
        device_x (int): Index of the x stage on the port (default 0).
        device_y (int): Index of the y stage on the port (default 1 if sharing, else 0).
        protocol (str): "ascii" or "binary" (default "ascii").

    Example usage:
    See openwfs\\examples\\Zaber translation stage control.ipynb for
    more elaborate example usage.

    # Both stages on same COM port
    stage_xy = ZaberXYStage("COM3")

    # Stages on different COM ports
    stage_xy = ZaberXYStage("COM3", "COM4")

    # Explicit device indices (if multiple devices per port)
    stage_xy = ZaberXYStage("COM3", device_x=0, device_y=1)

    # Move X and Y axes
    print("Current positions:", stage_xy.x, stage_xy.y)
    stage_xy.x = 5000 * u.um
    stage_xy.y = 3000 * u.um
    print("New positions:", stage_xy.x, stage_xy.y)

    # Home both axes
    stage_xy.home()
    print("Homed positions:", stage_xy.x, stage_xy.y)

    # Clean up / close connections
    stage_xy.close()
    """

    def __init__(
        self,
        port_x: str,
        port_y: str | None = None,
        device_x: int = 0,
        device_y: int | None = None,
        protocol: str = "ascii",
    ):
        # Initialize base classes
        # SerialPortBase.__init__(self, protocol)
        # Actuator.__init__(self, duration=0 * u.ms, latency=0 * u.ms)

        # If only one port is given, use it for both axes
        if port_y is None:
            port_y = port_x

        self.port_x = port_x  # save ports
        self.port_y = port_y
        self._owned_ports = [self.port_x, self.port_y]

        self.stage_x = ZaberLinearStage(port_x, device=device_x, protocol=protocol)
        self.stage_y = ZaberLinearStage(
            port_y, device=device_y if device_y is not None else (1 if port_y == port_x else 0), protocol=protocol
        )

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

    def close(self):
        self.stage_x.close()
        self.stage_y.close()


class ZaberLinearStage(Actuator, SerialPortBase):
    """
    Wraps a single Zaber stage so it looks like a Linear stage.
    Handles connection setup given a COM port.

    Args:
        port (str): COM port name for the stage (e.g. "COM3").
        device (int): Index of the stage on the port (default 0).

    Example usage:
    See openwfs\\examples\\Zaber translation stage control.ipynb for
    more elaborate example usage.

    # Create a linear stage on COM3, first device in chain
    stage_lin = ZaberLinearStage("COM3")

    # If multiple devices on COM3, pick the second one
    stage_lin = ZaberLinearStage("COM3", device=1)

    # Move and home
    print("Current pos:", stage_lin.x)
    stage_lin.x = 5000 * u.um
    print("New pos:", stage_lin.x)
    stage_lin.home()

    # Clean up
    stage_lin.close()
    """

    def __init__(self, port: str, device: int = 0, protocol: str = "ascii"):
        # Initialize base classes
        SerialPortBase.__init__(self, protocol)
        Actuator.__init__(self, duration=0 * u.ms, latency=0 * u.ms)

        self.port = port  # save port
        self._owned_ports = [self.port]

        # Open connection using base class helper
        self.connection = self.open_connection(port)

        # Detect devices
        devices = self.connection.detect_devices()

        # Pick the device
        self.stage = devices[device]

    @property
    def x(self) -> Quantity[u.um]:
        return self.stage.get_position(unit=Units.LENGTH_MICROMETRES) * u.um

    @x.setter
    def x(self, value: Quantity[u.um]):
        self.stage.move_absolute(value, Units.LENGTH_MICROMETRES)
        return self.x

    def home(self):
        self.stage.home()
        return self.x
