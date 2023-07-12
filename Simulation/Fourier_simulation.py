from SSA import SSA
from WFS import WFS
from Fourier import FourierDualRef
from Simulation import SimulatedWFS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from PIL import Image


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
#correct_wf = np.round(np.random.rand(4,5)*256)
#correct_wf = np.array([[0, 0],[128,128]])

# Load a sample image from scikit-image
test_image = data.camera()



#test_image = np.ceil((np.array(Image.open('bmpi2.png').convert('L')).astype(np.uint8))/64)*64+64




# Convert the image to a NumPy array
correct_wf = np.array(test_image)
# correct_wf = np.ones([10,10])*50
Sim.set_ideal_wf(correct_wf)


[feedback_set, ideal_wavefront, t_set] = WFS(Sim,feedback,FourierDualRef(3,np.zeros([1000,1000]),np.arange(-10,10,1),np.arange(-10,10,1),0.1,0))

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

plt.plot(feedback_set[:,:,0])
plt.legend([str(x) for x in correct_wf.flatten().tolist()])
plt.title('Feedback signals of the phase modulation of the SLM fields')
plt.show(block=True)