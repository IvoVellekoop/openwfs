import threading
import time
import weakref

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from serial.tools import list_ports
from zaber_motion import Units, ascii, binary

from ..core import Actuator


class _SerialPortConnection:
    """Wrapper for a serial port connection managed by the zaber_motion library.

    This wrapper automatically closes the connection when it is deleted.
    Note: this may not be needed if Zaber already does this automatically,
        but that behavior is not documented.
    Note: do not instantiate this class directly.
        Instead use `_SerialPortConnection.open` to open a connection,
        and re-use an existing connection if possible.
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
        stage.home()  # home both axes
        stage.x = 5000 * u.um  # move X to 5000 micrometers
        stage.y = 2 * u.mm  # move Y to 2 millimeters
        print(stage.x, stage.y)  # get current positions
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
        super().__init__(duration=np.inf * u.ms, latency=0 * u.ms)

        # If only one port is given, use it for both axes
        if port_y is None:
            port_y = port_x

        # If only one port is given and device_number_y is not specified, assume second device on same port and set device_number_y=1,
        # otherwise set device_number_y=0
        if device_number_y is None:
            device_number_y = 1 if port_y == port_x else 0

        # Create stage objects
        self.stage_x = ZaberLinearStage(port_x, device_number=device_number_x, protocol=protocol)
        self.stage_y = ZaberLinearStage(port_y, device_number=device_number_y, protocol=protocol)

    @property
    def x(self) -> Quantity[u.um]:
        return self.stage_x.position

    @x.setter
    def x(self, value: Quantity[u.um]):
        self.stage_x.position = value

    @property
    def y(self) -> Quantity[u.um]:
        return self.stage_y.position

    @y.setter
    def y(self, value: Quantity[u.um]):
        self.stage_y.position = value

    def home(self):
        self.stage_x.home()
        self.stage_y.home()

    def x_home(self):
        self.stage_x.home()

    def y_home(self):
        self.stage_y.home()

    @staticmethod
    def list_all_devices():
        """
        List all COM ports and devices connected to them using both ASCII and Binary protocols.
        Works even if some ports are already open by objects.

        Returns: dict { "COMx": [{"protocol": "ascii"|"binary", "devices": [...] }], ... }
        """
        return ZaberLinearStage.list_all_devices()


class ZaberLinearStage(Actuator):
    """
    Wrapper for a single Zaber linear stage connected via serial port.

    todo: asynchroneous behavior (now setting the position blocks until the stage has moved completely)
    todo: add a `settle_time` parameter to tell openwfs how long to wait after a move before starting a measurement.
    todo: at the moment it is possible to construct two stage objects referring to the same device. This condition
        should be checked and result in an error being raised.

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
        Actuator.__init__(self, duration=0 * u.ms, latency=0 * u.ms)

        # open a serial port connection and store the object to keep the connection alive
        self._serial_port = _SerialPortConnection.open(port, protocol)

        # store device number (for debugging only), and obtain an object for controlling the device
        self._device_number = device_number
        try:
            self._stage = self._serial_port.connection.detect_devices()[device_number]
        except IndexError:
            raise RuntimeError(
                f"No Zaber device found on port {port} at position {device_number} using {protocol} protocol."
            )

    @property
    def position(self) -> Quantity[u.um]:
        return self._stage.get_position(unit=Units.LENGTH_MICROMETRES) * u.um

    @position.setter
    def position(self, value: Quantity[u.um]):
        super()._start()  # wait for all previous commands to finish
        # Note: at the moment setting the stage position blocks until the stage has
        # moved completely. This should be changed so that 'busy' checks
        # if the movement has finished, but this behavior is not easily supported by
        # the binary Zaber API, which is needed for older stages.
        try:
            self._stage.move_absolute(value.to_value(u.um), Units.LENGTH_MICROMETRES)
        finally:
            # Set the end time to *now*. This tells the framework that it can start using
            # cameras and other detectors.
            # todo: here we should add a stabilization time to take into account
            #   vibrations after the motor stops.
            self._end_time_ns = time.time_ns()

    def home(self):
        self._stage.home()

    @staticmethod
    def list_all_devices():
        """
        List all COM ports and devices connected to them using both ASCII and Binary protocols.
        Works even if some ports are already open by objects.

        Returns: dict { "COMx": [{"protocol": "ascii"|"binary", "devices": [...] }], ... }
        """
        all_devices = {}
        for port in list_ports.comports():
            for protocol in ["ascii", "binary"]:
                try:
                    serial_port = _SerialPortConnection.open(port.device, protocol)
                except RuntimeError:
                    continue  # happens if the port is currently open using a different protocol.
                print(f"Port: {port}")
                try:
                    devices = serial_port.connection.detect_devices()
                    all_devices[port] = {"protocol": protocol, "devices": devices}
                    print(f"  Protocol: {protocol}")
                    for i, d in enumerate(devices):
                        print(f"    Device {i}: {d}")
                except Exception:
                    pass
                serial_port = None  # closes the port

        return all_devices
