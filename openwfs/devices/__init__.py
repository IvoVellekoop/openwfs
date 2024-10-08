try:
    from .camera import Camera
except ImportError:
    pass  # ok, we don't have harvesters installed

try:
    from . import galvo_scanner
    from .galvo_scanner import ScanningMicroscope, Axis
    from .nidaq_gain import Gain
except ImportError:
    pass  # ok, we don't have nidaqmx installed

from . import slm
from .slm import SLM
