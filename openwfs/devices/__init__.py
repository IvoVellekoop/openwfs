import warnings

try:
    import glfw
    import OpenGL
except ImportError:
    warnings.warn(
        """Could not initialize OpenGL because the glfw or PyOpenGL package is missing. 
        To install, make sure to install the required packages:
         ```pip install glfw```
         ```pip install PyOpenGL```
        Alternatively, specify the opengl extra when installing openwfs:
         ```pip install openwfs[opengl]```

        Note that these installs will fail if no suitable *OpenGL driver* is found on the system. 
        Please make sure you have the latest video drivers installed.
        """
    )
from . import slm
from .slm import SLM

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
