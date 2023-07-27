import sys
import os
# In order to run the examples as stand-alone scripts (main modules), they need to be able to find the openwfs package.
# The root of this package is two folders up from the current file, so we add that path to the search path.
p = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(p)


