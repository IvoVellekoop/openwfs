import ctypes
import weakref
from openwfs.devices import SLM
import numpy as np
import tempfile
import astropy.units as u


class BlinkHDMIHandler:
    """
    Class to handle the connection with the HDMI blink software. This class is used to ensure that the Blink software is properly initialized and closed.
    """

    def __init__(self):
        self.blink_lib = None
        self.path = None
        self.sdk_created = False

    def add_dll(self, path):
        """
        Add a file to the Blink software.
        :param file_path: The path to the file to be added.
        """
        if self.path is None:
            self.path = path
            ctypes.cdll.LoadLibrary(self.path)
            self.blink_lib = ctypes.CDLL("Blink_C_wrapper")
        else:
            if self.path != path:
                raise ValueError("A different DLL has already been loaded.")

        if not self.sdk_created:
            self.blink_lib.Create_SDK()
            self.blink_lib.Get_SLMTemp.restype = ctypes.c_double  # Taken from example file
            self.sdk_created = True

    @staticmethod
    def get_handler():
        global global_blinkhdmi_handler
        if type(global_blinkhdmi_handler) is weakref.ReferenceType:
            if global_blinkhdmi_handler is None:
                handler = BlinkHDMIHandler()
                global_blinkhdmi_handler = weakref.ref(handler)
            else:
                handler = global_blinkhdmi_handler()
        else:
            handler = BlinkHDMIHandler()
            global_blinkhdmi_handler = weakref.ref(handler)
        return handler

    def __del__(self):
        """
        Destructor for the BlinkHDMIHandler class. This method is called when the object is deleted and ensures that the Blink software is properly closed.
        """
        if self.sdk_created:
            self.blink_lib.Delete_SDK()
            self.sdk_created = False


global_blinkhdmi_handler = None


class SLMBlinkHDMI(SLM):

    def __init__(self, blink_path, lookup_table, slm_index=0, is_10bit=False, load_lookutp_table=True, **kwargs):
        self.handler = BlinkHDMIHandler.get_handler()
        self.handler.add_dll(blink_path)
        self.slm_blink_index = slm_index
        self.is_10bit = is_10bit
        self.bit_depth = 10 if self.is_10bit else 8

        self.usb_port = ctypes.create_unicode_buffer(256)
        status = self.handler.blink_lib.GetComPort(self.slm_blink_index, self.usb_port)
        if status == 0:
            raise RuntimeError(
                "SLM not found. The Blink SDK has a few issues. Check connections and restart python and try again (..and again probably...)"
            )

        if load_lookutp_table:
            self.load_lookup_table(lookup_table)

        super().__init__(**kwargs)

    def load_lookup_table(self, voltage_bits):
        """
        Load a lookup table into the Blink software.
        :param lut: The lookup table to be loaded.
        """
        # Create file
        # load file into blink software

        if voltage_bits.size != 2**self.bit_depth:
            raise ValueError(f"Lookup table must have {2**self.bit_depth} values for a {self.bit_depth}-bit SLM.")

        grey_bits = np.linspace(0, 2**self.bit_depth, num=voltage_bits.size, endpoint=False)

        voltage_bits = np.round(voltage_bits)
        data = np.column_stack((grey_bits, voltage_bits))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lut", delete=False) as f:
            np.savetxt(f, data, fmt="%d", delimiter="\t")
            filename = f.name

        status = self.handler.blink_lib.Load_lut(self.slm_blink_index, filename)
        if status == 0:
            raise RuntimeError("Loading the table on the SLM failed")

        self._lookup_table = voltage_bits

    @property
    def lookup_table(self):
        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, voltage_bits):
        self.load_lookup_table(voltage_bits)

    @property
    def temperature(self):
        return self.handler.blink_lib.Get_SLMTemp(self.slm_blink_index) * u.deg_C

    def linear_lookup_table(self):
        bit_grey = np.arange(2**self.bit_depth)
        bit_voltage = bit_grey * 4  # Map the 8/10 bit grey values to the 10/12 bit voltage value
        return bit_voltage
