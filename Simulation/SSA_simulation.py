from SSA import SSA
from WFS import WFS
from Simulation import SimulatedWFS
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

Sim = SimulatedWFS([500,500],active_plotting=True)

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
correct_wf = np.round(np.random.rand(8,8)*256)
#correct_wf = np.array([[0, 192],[25,22]])
# correct_wf = np.ones([10,10])*50
Sim.set_ideal_wf(correct_wf)


[feedback_set, ideal_wavefront, t_set] = WFS(Sim,feedback,SSA(16,np.zeros(np.shape(correct_wf))))

plt.figure(3)
plt.imshow(correct_wf)
plt.colorbar()
plt.clim(0, 256)
plt.title('Correct wavefront')
plt.figure(4)
Sim.set_data(correct_wf)
plt.title('Image for correct WF, feedback = '+"{:.2e}".format(feedback()[0]))

Sim.get_image()
plt.imshow(Sim.get_image())


plt.figure(5)
plt.imshow(ideal_wavefront)
plt.colorbar()
plt.clim(0, 256)
plt.title('Wavefront determined by SSA')

plt.figure(6)
Sim.set_data(ideal_wavefront)
Sim.get_image()
plt.title('Image for calculated wavefront, feedback = '"{:.2e}".format(feedback()[0]))
plt.imshow(Sim.get_image())

plt.figure(9)
Sim.set_data(0)
Sim.get_image()
plt.title('Image for flat wavefront, feedback = '"{:.2e}".format(feedback()[0]))
plt.imshow(Sim.get_image())

plt.figure(7)
diff = ideal_wavefront[:,:,0]-correct_wf
diff = np.where(diff < 0, 256 + diff, diff) % 256
plt.imshow(diff)
plt.colorbar()
plt.clim(0, 256)
# Sim.set_data(0)
# Sim.get_image()
plt.title('Difference between correct and ideal WF')
# plt.title('Image for flat wavefront, feedback = '"{:.2e}".format(feedback()[0]))
# plt.imshow(Sim.get_image())

plt.figure(8)

plt.plot(feedback_set)
plt.legend([str(x) for x in correct_wf.flatten().tolist()])
plt.title('Feedback signals of the phase modulation of the SLM fields')

plt.figure(9)
plt.imshow(np.log(np.reshape(np.std(feedback_set,axis=0),np.shape(correct_wf))))
plt.title('log of standard deviation of the phase modulation of the SLM fields')

plt.show(block=True)