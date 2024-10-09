import importlib
import warnings


def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        warnings.warn(
            f"""Could not import {module_name}, because the package is not installed.
            To install, use:
                pip install {module_name}
                
            Alternatively, specify to install the required extras when installing openwfs, using one of:
                pip install openwfs[all]
                pip install openwfs[opengl]
                pip install openwfs[genicam]
                pip install openwfs[nidaq]
            """
        )
        return None


from . import slm
from .slm import SLM
from .camera import Camera
from . import galvo_scanner
from .galvo_scanner import ScanningMicroscope, Axis
from .nidaq_gain import Gain
