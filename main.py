from slm import SLM
from patch import Patch

# construct a new SLM object and add a patch to it
s = SLM(0)
for n in range(10):
    s.update()

s.enumerate_monitors()