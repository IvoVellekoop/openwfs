"""
Camera live view
=====================
This script gives a _very_ rudimentary live viewer for a camera.
"""

import matplotlib.pyplot as plt
from harvesters.core import Harvester

h = Harvester()
h.add_file(R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti")
h.update()
ia = h.create(0)
ia.start()


def read():
    with ia.fetch() as buffer:
        component = buffer.payload.components[0]
        _1d = component.data
        _2d = component.data.reshape(component.height, component.width)
        return _2d.copy()


frame = 0
plt.figure()
im = plt.imshow(read())
plt.colorbar()
plt.show(block=False)
while True:
    im.set_data(read())
    plt.title(f"frame: {frame}")
    plt.show(block=False)
    plt.pause(0.001)
    frame += 1
