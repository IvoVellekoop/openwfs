from . import algorithms
from . import core
from . import devices
from . import processors
from . import simulation
from . import utilities
from .core import Detector, Device, Actuator, Processor, PhaseSLM

def _get_git_revision():
    """Check the current git revision of the code.
    Returns None if the git revision cannot be determined."""
    try:
        from pathlib import Path
        import subprocess

        try:
            path = Path(__file__).parent
        except NameError:
            path = Path.cwd()
        __current_commit_id__ = subprocess.check_output(["git", "describe", "--tags"], text=True, cwd=str(path)).strip()
    except Exception as e:
        return None
    return __current_commit_id__


try:
    from ._version import __version__, __commit_id__
except ImportError:
    import importlib.metadata as meta

    try:
        __version__ = meta.version("openwfs")
    except meta.PackageNotFoundError:
        # package is not installed
        __version__ = "<package not installed>"
    __commit_id__ = _get_git_revision()


def version():
    """Return the version information of the installed OpenWFS package.

    version is the version of the package during build. For development releases it will include the
        git hash and date of the commit. Note that the version is only updated during build (`uv build`), so it may be outdated if you
        are developing OpenWFS.
    commit_id is the git hash of the code that is currently used. If you have local commits but did not build the package yet,
        this value will differ from the tag in version.
    """
    current_id = _get_git_revision()
    if current_id is None:
        current_id = __commit_id__
    return dict(
        info="OpenWFS - © Ivo M. Vellekoop et al. University of Twente", version=__version__, commit_id=current_id
    )
