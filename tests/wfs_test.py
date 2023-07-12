from simulation.simulation import SimulatedWFS
from wfs_functions import WfsExperiment
from fourier import FourierDualRef

wfs = WfsExperiment()

wfs.algorithm = FourierDualRef()
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)

wfs.algorithm.kx_angles_stepsize = 2
wfs.algorithm.ky_angles_stepsize = 2
wfs.algorithm.ky_angles_max = 0
wfs.algorithm.build_kspace()
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)

wfs.algorithm.set_kspace([-4, 2], [7, 9])
print(wfs.algorithm.kx_set)
print(wfs.algorithm.ky_set)
wfs.slm_object = SimulatedWFS()
wfs.camera_object = SimulatedWFS()

wfs.execute = 1
# or you can use wfs.on_execute(), works either way