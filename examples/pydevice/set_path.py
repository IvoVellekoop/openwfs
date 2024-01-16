import sys
import os

# In order to run the examples as stand-alone scripts (main modules), they need to be able to find the openwfs package.
# The root of this package is in a sibling folder, so we add the parent folder to the search path.
# Note that PyTorch automatically does this, so you don't notice if it is left out.
p = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(p)
