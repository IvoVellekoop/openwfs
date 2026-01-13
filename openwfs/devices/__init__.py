import importlib
import sys
from types import ModuleType


class _MockModule(ModuleType):
    """A mock object that replaces a missing module, or a function / class from that module.

    This class makes it possible to run openwfs without all optional dependencies installed.
    If `safe_import` tries to load a package that is not installed, it creates a `MockModule`
    as placeholder for that package. The openwfs subpackage that tried to load this package
    will then load normally (also so that readthedocs can parse the docstrings).
    Only when a function from the missing package is called, an error is raised (see `__call__`)."""

    def __init__(self, module_name: str, extra_name: str):
        super().__init__(module_name)
        self.module_name = module_name
        self.extra_name = extra_name
        self.message = f"""Module {self.module_name} is not installed.
            To install, use:
                pip install openwfs[{self.extra_name}]
            or
                pip install openwfs[all]
            """

    def __getattr__(self, name):
        """Mimicks accessing a submodule, function or class in the module.

        This just returns another MockModule object. Only when this object is called (e.g. `glfw.init()`),
        an error is raised.
        """
        if name.startswith("_"):
            return super().__getattr__(name)
        else:
            return _MockModule(self.module_name, self.extra_name)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.message)


def is_loaded(module) -> bool:
    """Helper function to check if a module is a real loaded module or a mock module."""
    return not isinstance(module, _MockModule)


def safe_import(module_name: str, extra_name: str):
    """Tries to import a module.

    If the import files (usually because the module is not installed),
    this function registers a mock module so that importing can still continue without
    giving an error. This is essential for loading openwfs if optional dependencies are missing.
    This is also important for `readthedocs`, where not all dependenceis can be installed."""
    try:
        importlib.import_module(module_name)
    except (ModuleNotFoundError, AttributeError):
        sys.modules[module_name] = _MockModule(module_name, extra_name)


safe_import("harvesters", "genicam")
safe_import("harvesters.core", "genicam")
safe_import("nidaqmx", "nidaq")
safe_import("nidaqmx.system", "nidaq")
safe_import("nidaqmx.constants", "nidaq")
safe_import("nidaqmx.stream_writers", "nidaq")
safe_import("OpenGL", "opengl")
safe_import("OpenGL.GL", "opengl")
safe_import("glfw", "opengl")
safe_import("zaber_motion", "zaber")
safe_import("serial", "zaber")
safe_import("serial.tools", "zaber")
safe_import("clr", "clr")

from .camera import Camera
from .galvo_scanner import ScanningMicroscope, Axis
from .nidaq_gain import Gain
from . import slm
from .slm import SLM
from .zaber_stage import ZaberXYStage, ZaberLinearStage
from .kcube_inertial import KCubeInertialMotor
