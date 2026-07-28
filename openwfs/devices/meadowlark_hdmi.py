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
    """
    Class to control a Meadowlark SLM using the Blink software. The SLM uses the Blink software to mainly load lookup tables on the SLM which allows to achieve a higher bit depth than when using the software lookup table available on the openwfs SLM class. The SLM screen is still controlled using the openwfs SLM class.

    Args:
        blink_path (str): Path to the Blink DLL file.
        lookup_table (np.ndarray): Lookup table to be loaded on the SLM. (Or already loaded)
        slm_index (int, optional): Index of the SLM to be used. This index is the SLM index defined on Blink. Defaults to 0.
        is_10bit (bool, optional): Whether the SLM is 10-bit or not. Defaults to False.
        load_lookup_table (bool, optional): Whether to load the lookup table on initialization. Defaults to True. If False, the lookup table passed to the constructor must match the lookup table already loaded on the SLM.
    """

    def __init__(self, blink_path, lookup_table, slm_index=0, is_10bit=False, load_lookup_table=True, **kwargs):
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

        if load_lookup_table:
            self.load_lookup_table(lookup_table)

        default_encoding = {"encoding": "10b_rb" if is_10bit else "8b_r"}

        super().__init__(**(default_encoding | kwargs))

    def _create_lut_file(self, voltage_bits):
        """
        Create a lookup table temporary file to be uploaded to the SLM. The filename is returned.

        Args:
            voltage_bits: The lookup table to be loaded. The lookup table must have 2**bit_depth values, and tells how each grey value is mapped to the voltage value. The values of the lookup table must be in the range of 0 to 2**(bit_depth + 2) - 1. For example, for a 10-bit SLM, the values must be in the range of 0 to 4095.

        Returns:
            filename: The name of the file created. The file is created in a temporary directory and
        """
        if voltage_bits.size != 2**self.bit_depth:
            raise ValueError(f"Lookup table must have {2**self.bit_depth} values for a {self.bit_depth}-bit SLM.")

        grey_bits = np.linspace(0, 2**self.bit_depth, num=voltage_bits.size, endpoint=False)

        voltage_bits = np.round(voltage_bits)
        data = np.column_stack((grey_bits, voltage_bits))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lut", delete=False) as f:
            np.savetxt(f, data, fmt="%d", delimiter="\t")
            filename = f.name

        return filename

    def load_lookup_table(self, voltage_bits):
        """
        See the lookup_table property for more information on how to use this method.
        """
        # Create file
        # load file into blink software

        filename = self._create_lut_file(voltage_bits)

        status = self.handler.blink_lib.Load_lut(self.slm_blink_index, filename)
        if status == 0:
            raise RuntimeError("Loading the table on the SLM failed")

        self._lookup_table = voltage_bits

    def store_lookup_table(self):
        """
        Store the currently loaded lookup table on the SLM into the permanent memory of the SLM. This allows to keep the lookup table even after the SLM is turned off.
        """
        status = self.handler.blink_lib.Store_lut(self.slm_blink_index)
        if status == 0:
            raise RuntimeError("Storing the table on the SLM failed")

    @property
    def lookup_table(self):

        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, voltage_bits):
        """
        Load the lookup table on the SLM using the Blink software.

        Args:
            voltage_bits: The lookup table to be loaded. The lookup table must have 2**bit_depth values, and tells how each grey value is mapped to the voltage value. The values of the lookup table must be in the range of 0 to 2**(bit_depth + 2) - 1. For example, for a 10-bit SLM, the values must be in the range of 0 to 4095.
        """
        self.load_lookup_table(voltage_bits)

    @property
    def temperature(self):
        """
        Returns the temperature of the SLM in degrees Celsius. The temperature is read from the SLM using the Blink software.
        """
        return self.handler.blink_lib.Get_SLMTemp(self.slm_blink_index) * u.deg_C

    def linear_lookup_table(self):
        """
        Returns a linear lookup table for the SLM. The linear lookup table maps the grey values to the voltage values linearly. The voltage values are in the range of 0 to 2**(bit_depth + 2) - 1. For example, for a 10-bit SLM, the voltage values are in the range of 0 to 4095.
        """
        bit_grey = np.arange(2**self.bit_depth)
        bit_voltage = bit_grey * 4  # Map the 8/10 bit grey values to the 10/12 bit voltage value
        return bit_voltage
