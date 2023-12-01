from time import time, sleep
import matplotlib.pyplot as plt
from pycromanager import Core

core = Core()
dir(core)

print(core.get_property('Device:gain', 'Gain'))

core.set_property('Camera:scanner', 'Height', 250)
print(core.get_property('Camera:scanner', 'Height'))

t = time()
core.set_property('PIZStage', 'Position', 75)
print(time() - t)
sleep(0.5)

core.snap_image()
tagged_img1 = core.get_tagged_image()
plt.figure()
plt.imshow(tagged_img1.pix.reshape(tagged_img1.tags['Height'], tagged_img1.tags['Width']))

t = time()
core.set_property('PIZStage', 'Position', 225)
print(time() - t)
sleep(0.5)

core.snap_image()
tagged_img2 = core.get_tagged_image()
plt.figure()
plt.imshow(tagged_img2.pix.reshape(tagged_img2.tags['Height'], tagged_img2.tags['Width']))

plt.show()

pass
