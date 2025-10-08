import astropy.units as u
from astropy.units import Quantity
from zaber_motion import Units, Library
from zaber_motion import ascii, binary
import serial.tools.list_ports
from ..core import Actuator
import weakref
import threading


class SerialPortBase:
    """
    Base class for devices connected via serial ports, handles protocol selection,
    connection setup, device detection, and shared connections.

    Design:
      - Strong dict: port -> connection
      - WeakSet of owners per port
      - Finalizer on each owner to release ownership automatically
      - Single lock protects shared maps (thread-safe)

    General idea:
        Make sure that multiple objects using the same port share a single connection,
        and that the connection is closed automatically when the last object is deleted.
        If an object is overwritten or goes out of scope, the connection is closed if no other
        objects are using it.

    Uses some specifics of Zaber Motion Library connections and devices.
    Could possibly be adapted in the future for general serial devices.
    """

    _connections = {}  # {port: connection} strong refs
    _owners = {}  # {port: weakref.WeakSet[SerialPortBase]}
    _lock = threading.RLock()

    def __init__(self, port: str, device_number: int = 0, protocol: str = "ascii"):
        self.protocol = protocol.lower()
        if self.protocol == "ascii":
            self.ConnectionClass = ascii.Connection
        elif self.protocol == "binary":
            self.ConnectionClass = binary.Connection
        else:
            raise ValueError("protocol must be 'ascii' or 'binary'")

        self.port = port

        # Open/reuse connection and register ownership under the lock.
        with SerialPortBase._lock:
            conn = SerialPortBase._connections.get(port)
            if conn is None:
                conn = self.ConnectionClass.open_serial_port(port)
                SerialPortBase._connections[port] = conn
                SerialPortBase._owners[port] = weakref.WeakSet()

            # Track this owner (weakly to allow GC)
            SerialPortBase._owners[port].add(self)

            # Finalizer on the OWNER, not the connection; don't capture conn.
            self._owner_finalizer = weakref.finalize(
                self,
                SerialPortBase._release_owner,
                port,
                weakref.ref(self),
            )
            self.connection = conn

        # Detect devices (slow operations outside the lock)
        try:
            devices = self.connection.detect_devices()
            if not devices:
                raise RuntimeError(f"No Zaber devices found on port {port} using {protocol} protocol.")
            self.device = devices[device_number]
        except Exception:
            # Undo ownership if init fails, for example if device_number is out of range
            self._safe_release_owner()
            raise

    @staticmethod
    def _release_owner(port: str, self_ref: "weakref.ReferenceType"):
        """
        Called when an owner is GC'ed.
        Removes the owner; if no *live* owners remain, closes and removes the connection.
        """
        with SerialPortBase._lock:
            owners = SerialPortBase._owners.get(port)
            if owners is None:
                return  # Already cleaned

            # Try to remove this specific owner if still alive
            obj = self_ref()
            if obj is not None:
                owners.discard(obj)

            # IMPORTANT: Force WeakSet to clean up dead refs before deciding
            has_live_owner = False
            for _ in owners:  # iteration triggers internal cleanup of dead weakrefs
                has_live_owner = True
                break

            if not has_live_owner:
                # Last live owner gone -> close and cleanup
                conn = SerialPortBase._connections.pop(port, None)
                SerialPortBase._owners.pop(port, None)
                if conn is not None:
                    try:
                        print(f"[AUTO-CLOSE] Last owner gone for {port} — closing.")
                        conn.close()
                    except Exception as e:
                        print(f"[WARN] Auto-close failed for {port}: {e}")

    def _safe_release_owner(self):
        """Invoke the owner finalizer exactly once."""
        fin = getattr(self, "_owner_finalizer", None)
        if fin and fin.alive:
            fin()

    # ToDo: decide if we want a manual close() method
    # A starts has been made below, but there are some bugs (ports stay open if finalizer already ran)

    # def close(self):
    #     """
    #     Manual close: release this owner.
    #     The port is closed automatically when the last owner releases.
    #     """
    #     self._safe_release_owner()
    #     self.connection = None

    # def close(self):
    #     """
    #     Manual close: release this owner; closes the port when the last owner releases.
    #     Works even if the finalizer already ran or wasn't registered yet.
    #     """
    #     # Preferred path: run the finalizer (idempotent; runs at most once)
    #     fin = getattr(self, "_owner_finalizer", None)
    #     if fin and fin.alive:
    #         fin()  # calls _release_owner under the class lock
    #     else:
    #         # Fallback: ensure we are removed from owners and close if last
    #         with SerialPortBase._lock:
    #             owners = SerialPortBase._owners.get(self.port)
    #             if owners is not None:
    #                 owners.discard(self)

    #                 # Purge and check for live owners
    #                 has_live_owner = False
    #                 for _ in owners:
    #                     has_live_owner = True
    #                     break

    #                 if not has_live_owner:
    #                     conn = SerialPortBase._connections.pop(self.port, None)
    #                     SerialPortBase._owners.pop(self.port, None)
    #                     if conn is not None:
    #                         try:
    #                             print(f"[CLOSE] Last owner gone for {self.port} — closing.")
    #                             conn.close()
    #                         except Exception as e:
    #                             print(f"[WARN] Error closing {self.port}: {e}")

    #     # Clear instance handle to avoid accidental reuse
    #     self.connection = None

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

            # If port is already open, query via existing connection (no lock during I/O)
            with SerialPortBase._lock:
                conn = SerialPortBase._connections.get(port.device)

            if conn is not None:
                try:
                    devices = conn.detect_devices()
                    if devices:
                        proto_name = "ascii" if isinstance(conn, ascii.Connection) else "binary"
                        port_info.append({"protocol": proto_name, "devices": devices})
                except Exception as e:
                    print(f"[WARN] Could not query existing connection {port.device}: {e}")
            else:
                # Probe with temporary connections; close them immediately.
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

        # Pretty-print
        for p, info_list in devices_found.items():
            print(f"Port: {p}")
            for info in info_list:
                proto = info["protocol"]
                devs = info["devices"]
                print(f"  Protocol: {proto}")
                for i, d in enumerate(devs):
                    print(f"    Device {i}: {d}")

        return devices_found

    @staticmethod
    def list_open_ports():
        with SerialPortBase._lock:
            return list(SerialPortBase._connections.keys())

    # --- Debug helpers -------------------------------------------------------

    @classmethod
    def debug_owner_details(cls) -> dict:
        """
        Snapshot with lightweight owner details. Structure:
        {
            "COM4": {
                "count": 2,
                "owner_ids": ["0x108f8c2a0", "0x108f8d1e0"],
                "owner_types": ["SerialPortBase", "SerialPortBase"],
                "connection": "repr-of-connection"
            },
            ...
        }
        Note: only returns strings/ints (no strong refs to owners).
        """
        out = {}
        with cls._lock:
            for port, conn in cls._connections.items():
                owners = cls._owners.get(port)
                if owners is None:
                    out[port] = {
                        "count": 0,
                        "owner_ids": [],
                        "owner_types": [],
                        "connection": repr(conn),
                    }
                    continue

                ids = []
                types = []
                # Create a temporary list to avoid set-size changes during iteration
                for o in list(owners):
                    if o is None:
                        continue
                    ids.append(hex(id(o)))
                    types.append(type(o).__name__)

                out[port] = {
                    "count": len(ids),
                    "owner_ids": ids,
                    "owner_types": types,
                    "connection": repr(conn),
                }
        return out

    @classmethod
    def debug_print_state(cls, include_owner_ids: bool = False, include_conn_repr: bool = False):
        """
        Pretty-print current connection pool and owners.
        """
        with cls._lock:
            if not cls._connections:
                print("No open connections.")
                return

            print("=== SerialPortBase debug state ===")
            for port, conn in cls._connections.items():
                owners = cls._owners.get(port)
                count = len(owners) if owners is not None else 0
                print(f"Port {port}: owners={count}")
                if include_conn_repr:
                    print(f"  conn={conn!r}")
                if include_owner_ids and owners:
                    for o in list(owners):
                        if o is None:
                            continue
                        print(f"   - owner id={hex(id(o))}, type={type(o).__name__}")
            print("=== end ===")


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
        return SerialPortBase.list_all_devices()


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
        self.serial_port = SerialPortBase(port, device_number=device_number, protocol=protocol)
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
        return SerialPortBase.list_all_devices()
