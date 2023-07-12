import sys
from pathlib import Path
import time

# Get the directory of the current file
current_dir = Path(__file__).resolve().parent

# Construct the relative path
slm_path = current_dir.parent / 'hardware' / 'generic_binding'
python_binding_path = current_dir.parent / 'micro-manager' / 'mmCoreAndDevices' / 'DeviceAdapters' / 'MM_pydevice'

# Add the relative path to sys.path
sys.path.append(str(slm_path))
sys.path.append(str(python_binding_path))


from Pyscanner import single_capture
from SLMwrapper import SLM, set_circular_geometry, test

from SSA import SSA
from WFS import WFS
from Fourier import FourierDualRef

s = SLM(0)
import numpy as np

def single_line():
    im = single_capture(resolution=[10,10])
    return im[:,0]

count = 0


def single_point():
    global count

    result = count % 7
    count += 1
    # test if the algorithm is able to optimise despite 3 non-signal feedback points
    # return [result, np.round(np.random.rand()*20), np.round(np.random.rand()*20), np.round(np.random.rand()*20)]
    return [result]

#[feedback_set, ideal_wavefronts, t_set] = WFS(s,single_point,SSA(20,np.zeros([4,4])))




init_wf = np.zeros([100,100])
kx = np.arange(-2,2,2)
ky = np.arange(-2,2,2)
overlap_coef = 0.1


s.set_data(ideal_wavefronts[:,:,0])

s.update()
time.sleep(10)
