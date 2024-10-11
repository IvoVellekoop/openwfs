import importlib
import warnings


def safe_import(module_name: str, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, AttributeError):
        warnings.warn(
            f"""Could not import {module_name}, because the package is not installed.
            To install, using:
                pip install openwfs[{extra_name}]
            or
                pip install openwfs[all]
            """
        )
        return None


from . import slm
from .slm import SLM
from .camera import Camera
from . import galvo_scanner
from .galvo_scanner import ScanningMicroscope, Axis
from .nidaq_gain import Gain
