from ssa import StepwiseSequential
from wfs import wavefront_shaping
from openwfs.simulation import SimulatedWFS
import numpy as np

Sim = SimulatedWFS()
Sim.E_input_slm = np.ones([500,500])
def get_center(Simulation):
    Simulation.get_image()
    def get_center_closure():
        Simulation.trigger()
        Simulation.wait()
        im = Simulation.image
        shape = np.shape(im)
        return [im[int(np.ceil(shape[0]/2)),int(np.ceil(shape[1]/2))]]

    return get_center_closure

feedback = get_center(Sim)
n = 4
correct_wf = np.round(np.random.rand(n,n)*256)
#correct_wf = np.array([[0, 192],[25,22]])
# correct_wf = np.ones([10,10])*50
Sim.set_ideal_wf(correct_wf)

ssa = StepwiseSequential()
ssa.n_slm_fields = n
[feedback_set, ideal_wavefront, t_set] = wavefront_shaping(Sim,feedback,ssa)