from openwfs.algorithms import StepwiseSequential
from wfs import wavefront_shaping
from openwfs.algorithms import FourierDualRef
from openwfs.simulation import SimulatedWFS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from PIL import Image


Sim = SimulatedWFS()
Sim.E_input_slm = np.ones([500, 500])


def get_center(Simulation):
    Simulation.get_image()

    def get_center_closure():
        Simulation.trigger()
        Simulation.wait()
        im = Simulation.image
        shape = np.shape(im)
        return [im[int(np.ceil(shape[0]/2)), int(np.ceil(shape[1]/2))]]

    return get_center_closure


feedback = get_center(Sim)

# Load a sample image from scikit-image
test_image = data.camera()

# Convert the image to a NumPy array
correct_wf = np.array(test_image)
# correct_wf = np.ones([10,10])*50
Sim.set_ideal_wf(correct_wf)
alg = FourierDualRef()

[feedback_set, ideal_wavefront, t_set] = wavefront_shaping(Sim, feedback, alg)

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
plt.colorbar()
plt.clim(0, 2**16)

plt.figure(5)
plt.imshow(ideal_wavefront)
plt.colorbar()
plt.clim(0, 256)
plt.title('Wavefront determined by Fourier')

plt.figure(6)
Sim.set_data(ideal_wavefront)
Sim.get_image()
plt.title('Image for calculated wavefront, feedback = '"{:.2e}".format(feedback()[0]))

plt.imshow(Sim.get_image())
plt.colorbar()
plt.clim(0, 2**16)

plt.figure(7)
Sim.set_data(0)
Sim.get_image()
plt.title('Image for flat wavefront, feedback = '"{:.2e}".format(feedback()[0]))

plt.imshow(Sim.get_image())
plt.colorbar()
plt.clim(0, 2**16)


plt.figure(8)

plt.plot(feedback_set[:, :, 0])
plt.legend([str(x) for x in correct_wf.flatten().tolist()])
plt.title('Feedback signals of the phase modulation of the SLM fields')
plt.show(block=True)
