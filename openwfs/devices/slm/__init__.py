try:
    import glfw
    import OpenGL
except ImportError:
    raise ImportError(
        """Could not initialize OpenGL because the glfw or PyOpenGL package is missing. 
        To install, make sure to install the required packages:
         ```pip install glfw```
         ```pip install PyOpenGL```
        Alternatively, specify the opengl extra when installing openwfs:
         ```pip install openwfs[opengl]```
         
        Note that these installs will fail if no suitable OpenGL driver is found on the system. 
        Please make sure you have the latest video drivers installed.
        """
    )

from . import geometry
from . import patch
from . import shaders
from . import slm
from .geometry import Geometry, circular, rectangle
from .patch import Patch
from .slm import SLM
