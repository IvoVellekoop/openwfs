import sys
import os

os.chdir('C:/Users/Jeroen Doornbos/Documents/wfs_current/wavefront_shaping_python')

# SLM
sys.path.append('../hardware/generic_binding')
# Scanner
sys.path.append('../micro-manager/mmCoreAndDevices/DeviceAdapters/MM_pydevice')

from Pyscanner import single_capture
from SLMwrapper import SLM, set_circular_geometry, test


import matplotlib.pyplot as plt
import time
import numpy as np

plt.imshow(single_capture())
s = SLM(0)
int_array = np.round((single_capture(resolution = [100,100]) / 2**16) * 255).astype(np.uint8)
s.set_data(int_array)
s.update()
time.sleep(2)