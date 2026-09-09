import ctypes
import weakref
from typing import Optional
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

    def add_dll(self, path: str) -> None:
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
    def get_handler(path: str) -> "BlinkHDMIHandler":
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
        handler.add_dll(path)
        return handler

    def __del__(self) -> None:
        """
        Destructor for the BlinkHDMIHandler class. This method is called when the object is deleted and ensures that the Blink software is properly closed.
        """
        if self.sdk_created:
            self.blink_lib.Delete_SDK()
            self.sdk_created = False


global_blinkhdmi_handler = None


class SLMBlinkHDMI(SLM):
    """
    Class to control a Meadowlark SLM using the Blink software. The SLMBlinkHDMI has 2 different lookup tables namely hardware_lookup_table and lookup_table. The hardware_lookup_table operates within the SLM and maps the screen image to the voltage DAC values of the SLM screen. The lookup_table is the fast lookup_table available within the openwfs SLM class. 

    Args:
        blink_path: Path to the Blink DLL file.
        hardware_lookup_table: Lookup table to be loaded on the hardware of the SLM. (Or pre-loaded if the load_lookup_table is set to False)
        slm_index: Index of the SLM to be used. This index is the SLM index defined on Blink. Defaults to 0.
        load_lookup_table: Whether to load the hardware lookup table on initialization. Defaults to True. If False, the lookup table used will be the lookup table previously loaded on the slm. For correctness, the hardware_lookup_table passed to the constructor must match the lookup table loaded on the memory of the  SLM. If you are unsure, always set _load_lookup_table to True.
        **kwargs: Additional keyword arguments to be passed to the SLM class. The default value of enconding is set to "10b_rb" if the SLM is 10-bit and "8b_r" if the SLM is 8-bit. This can be overridden by passing an encoding argument in kwargs.
    """

    def __init__(self, blink_path: str, hardware_lookup_table: np.ndarray, slm_index: int = 0, load_hardware_lookup_table: bool = True, **kwargs) -> None:
        self.handler = BlinkHDMIHandler.get_handler(blink_path)
        self.slm_blink_index = slm_index

        str_usb_port = ctypes.create_unicode_buffer(256)
        status = self.handler.blink_lib.GetComPort(self.slm_blink_index, str_usb_port)
        self.usb_port = str_usb_port.value

        bit_depth = self.handler.blink_lib.Get_Depth(self.slm_blink_index)

        if status == 0:
            raise RuntimeError(
                "SLM not found. The Blink SDK has a few issues. Check connections and restart python and try again (..and again probably...). A common issue is the corrupted Preferences.ini file in the Blink software folder. Try reseting the Preferences.ini file to the settings of a new installation."
            )

        if load_hardware_lookup_table:
            self._load_lookup_table(hardware_lookup_table)
        else:
            self._hardware_lookup_table = hardware_lookup_table

        default_encoding = {"encoding": "10b_rb" if bit_depth==10 else "8b_r"}

        super().__init__(**(default_encoding | kwargs))

    def _create_lut_file(self, voltage_bits: np.ndarray) -> str:
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

    def _load_lookup_table(self, voltage_bits: np.ndarray) -> None:
        """
        See the hardware_lookup_table property for more information on how to use this method.
        """
        # Create file
        # load file into blink software
        filename = self._create_lut_file(voltage_bits)

        status = self.handler.blink_lib.Load_lut(self.slm_blink_index, filename)
        if status == 0:
            raise RuntimeError("Loading the table on the SLM failed")

        self._lookup_table = voltage_bits
    
    @property
    def hardware_lookup_table(self) -> np.ndarray:
        return self._hardware_lookup_table

    @hardware_lookup_table.setter
    def hardware_lookup_table(self, voltage_bits: np.ndarray, to_permament_memory: bool = False) -> None:
        """
        Load a lookup table on the SLM using the Blink software. This lookup table is unloaded when the SLM is turned off. If to_permament_memory is set to True, the lookup table will be stored in the permanent memory of the SLM and will be kept even after the SLM is turned off.

        Args:
            voltage_bits: The lookup table to be loaded. The lookup table must have 2**bit_depth values, and tells how each grey value is mapped to the voltage value. The values of the lookup table must be in the range of 0 to 2**(bit_depth + 2) - 1. For example, for a 10-bit SLM, the values must be in the range of 0 to 4095. For example to load a linear lookup table, voltage_bits = np.arange(2**slm.bit_depth) * 4.
        """
        self._load_lookup_table(voltage_bits)
        if to_permament_memory:
            self._store_lookup_table()
            status = self.handler.blink_lib.Store_lut(self.slm_blink_index)
            if status == 0:
                raise RuntimeError("Storing the table on the SLM failed")
        self._hardware_lookup_table = voltage_bits

    @property
    def temperature(self) -> u.Quantity[u.deg_C]:
        """
        Returns the temperature of the SLM in degrees Celsius. The temperature is read from the SLM using the Blink software.
        """
        return self.handler.blink_lib.Get_SLMTemp(self.slm_blink_index) * u.deg_C

    def get_lookup_table_filename(self) -> str:
        """
        Returns the filename of the lookup table currently loaded on the SLM.
        """

        filename = ctypes.create_unicode_buffer(256)
        status = self.handler.blink_lib.GetLUTFileName(self.slm_blink_index, filename)

        if status == 0:
            raise RuntimeError("Getting the filename of the lookup table failed")

        return filename.value
