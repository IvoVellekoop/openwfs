from slm import SLM
from patch import Patch

# construct a new SLM object and add a patch to it
s = SLM(0)
rectangle = Patch()
s.patches.append(rectangle)
for n in range(10):
    s.update()

s.enumerate_monitors()