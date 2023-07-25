import sys
import os
p = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(p)

# Import this module to add the parent folder to the search path.
# This done by the examples in `simulation` so that they can be run as stand-alone scripts (main modules) and still use
# the files in the `slm` and other packages.

