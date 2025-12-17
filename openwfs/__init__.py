from . import algorithms
from . import core
from . import devices
from . import processors
from . import simulation
from . import utilities
from .core import Detector, Device, Actuator, Processor, PhaseSLM

try:
    from ._version import __version__, __commit_id__
except ImportError:
    import importlib.metadata as meta
    import subprocess

    try:
        __version__ = meta.version("openwfs")
    except meta.PackageNotFoundError:
        # package is not installed
        __version__ = "<package not installed>"
    try:
        __commit_id__ = subprocess.check_output(["git", "describe", "--tags"], text=True, cwd=str(root_path)).strip()
    except Exception as e:
        __commit_id__ = f"unknown ({e})"


def version():
    """Return the version string of the installed OpenWFS package.

    This includes __version__ and __git_revision__.
    __version__ is the version of the installed package, as reported by the package metadata.
    __git_revision__ is a string composed of the latest tag found in the code branch + the number of commits since that tag + the hash of the current commit.
    """

    return f"OpenWFS {__version__} ({__commit_id__})"
