from SSA import SSA
from WFS import WFS
from Simulation import SimulatedWFS
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

Sim = SimulatedWFS([500,500],active_plotting=False)

def get_center(Simulation):

    def get_center_closure():
        im = Simulation.get_image()
        shape = np.shape(im)
        return [im[int(np.ceil(shape[0]/2)),int(np.ceil(shape[1]/2))]]

    return get_center_closure

feedback = get_center(Sim)
correct_wf = np.round(np.random.rand(4,5)*256)
#correct_wf = np.array([[0, 192],[25,22]])
# correct_wf = np.ones([10,10])*50
Sim.set_ideal_wf(correct_wf)

def field_correlation(E1,E2):
    E1_norm = E1/ np.linalg.norm(E1)
    E2_norm = E2/ np.linalg.norm(E2)
    product = np.conj(E1_norm) * E2_norm

    return np.sum(product)

correlations = np.array([])
import numpy as np
import time
from scipy.sparse import lil_matrix

import numpy as np
import time
from scipy.sparse import lil_matrix

max_SLM_elements = 30

all_corr = [[] for _ in range(max_SLM_elements - 2)]

for i in range(2, max_SLM_elements):
    start_time = time.time()
    correlations = []
    print(i)
    while time.time() - start_time < 10:  # Run for 20 seconds
        correct_wf = np.round(np.random.rand(1, i) * 256)
        Sim.set_ideal_wf(correct_wf)
        [feedback_set, ideal_wavefront, t_set] = WFS(Sim, feedback, SSA(3, np.zeros([1, i])))
        field_corr = field_correlation(np.exp(-1j * correct_wf / (256 / (2 * np.pi))),
                                       np.exp(-1j * ideal_wavefront[:, :, 0] / (256 / (2 * np.pi))))

        correlations.append(abs(field_corr))

    all_corr[i - 2]= [correlations]

# Plotting
x_values = np.arange(2, max_SLM_elements)
means = np.array([])
for i, correlations in enumerate(all_corr):
    y_values = np.squeeze(correlations)
    x_indices = np.nonzero(correlations)[0] + 2 +i
    if i == 0:
        plt.plot(x_indices, y_values, 'bo',label='Field correlation individual experiments')
    else:
        plt.plot(x_indices, y_values, 'bo')

    means = np.append(means,np.mean(correlations))

plt.plot(np.arange(2,2+len(means),1),means,'r',label='Mean field correlation')
plt.xlabel("SLM Elements")
plt.ylabel("Correlation")
plt.title("Correlation vs SLM Elements")
plt.legend()
plt.show()



