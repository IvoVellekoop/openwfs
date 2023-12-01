#!/usr/bin/env python
# coding: utf-8

# In[4]:


import set_path


import numpy as np
from openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi, CropProcessor, SingleRoiSquare, SelectRoiSquare
from openwfs.devices import LaserScanning, Gain
from openwfs.slm import SLM
from openwfs.slm.geometry import fill_transform
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

scanner = LaserScanning(x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3', input_mapping='Dev4/ai24',measurement_time=600 * u.ms,
                       height = 512,
                       width = 512)

gain = Gain()
gain.on_reset(1)


# In[3]:


gain.on_reset(1)


# In[4]:


gain.gain = 0.65


# In[ ]:


gain.gain = 0.9
scanner.zoom = 80
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.measurement_time = 200 * u.ms
scanner.trigger()

plt.imshow(scanner.read())
plt.colorbar()


# In[35]:


gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()

plt.imshow(scanner.read(), vmin=32700, vmax=33000)
plt.colorbar()


# In[30]:


square_detector = SelectRoiSquare(scanner)
scanner.trigger()
plt.imshow(square_detector.read_square())
plt.colorbar()

square_detector.read()


# In[68]:


square_detector.left


# In[69]:


square_detector.top


# In[70]:


square_detector.width


# In[71]:


square_detector.height


# In[31]:


slm = SLM(2)

# hardcode offset, because our calibrations don't work yet
transform_matrix = np.array(fill_transform(slm,type='short'))
transform_matrix = transform_matrix*0.8 #scaling according to last
transform_matrix[2,:] = [-0.0147/(0.4+0.5), 0.0036/0.5, 1] # from the old hardcoded offset, visually adjusted to be right

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33)/255 # again copied from earlier hardcodes
slm.wavelength = 0.720 * u.um

slm.transform = transform_matrix
controller = Controller(detector=square_detector, slm=slm)


# In[ ]:





# In[32]:


import matplotlib.pyplot as plt
import hdf5storage as h5

# Load data
patterndict = h5.loadmat(file_name="//ad.utwente.nl/tnw/BMPI/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-top-λ808.0nm.mat")

# Plot
# plt.imshow(np.angle(patterndict['field_SLM']))
# plt.title('Field angle')
# plt.colorbar()
# plt.show()

# plt.imshow(patterndict['phase_SLM'] * 255/(2*np.pi), vmin=-1200, vmax=1000)
# plt.title('Gray value unwrapped')
# plt.colorbar()
# plt.show()


# In[33]:


from openwfs.slm import Patch
pf = Patch(slm)


# In[34]:


pf.phases = -patterndict['phase_SLM']
slm.update()


# In[ ]:


slm.update()


# In[36]:


alg_basic = BasicFDR(k_angles_min=-3, k_angles_max=3, phase_steps=8, overlap=0.1, controller=controller)
t_basic = alg_basic.execute()


# In[39]:


plt.imshow(abs(np.reshape(alg_basic.t_left,[7,7])))

plt.figure()
plt.imshow(abs(np.reshape(alg_basic.t_right,[7,7])))


# In[40]:


plt.imshow(np.angle(t_basic))


# In[41]:


gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()

plt.imshow(scanner.read(), vmin=32700, vmax=33000)
plt.colorbar()


# In[42]:


slm.phases = 0
slm.update()


# In[59]:


slm.phases = 0
pf.phases = -patterndict['phase_SLM']
slm.update()

gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()
before_im_basic = scanner.read().copy()

plt.imshow(before_im_basic, vmin=32700, vmax=33500)
plt.colorbar()


# In[60]:


slm.phases = np.angle(t_basic)
slm.update()
gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()
after_im_basic = scanner.read().copy()

plt.imshow(after_im_basic, vmin=32700, vmax=33500)
plt.colorbar()


# In[62]:


slm.phases = 0
pf.phases = 0
slm.update()
gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()
flat_im_basic = scanner.read().copy()

plt.imshow(flat_im_basic, vmin=32700, vmax=33500)
plt.colorbar()


# In[64]:


slm.phases = 0
pf.phases = 0
slm.update()
gain.gain = 0.7
scanner.zoom = 4
scanner.delay = 65
scanner.width = 200
scanner.height = 200
scanner.bidirectional = False
scanner.invert = True
scanner.dwelltime = 5 * u.us
scanner.trigger()
background_im_basic = scanner.read().copy()

plt.imshow(background_im_basic, vmin=32700, vmax=33500)
plt.colorbar()


# In[63]:


np.save('before_wfs_with_model',before_im_basic)
np.save('after_wfs_with_model',after_im_basic)
np.save('before_wfs_without_model',flat_im_basic)
np.save('t_matrix_slm',t_basic)
np.save('t_matrix_fourier_left',alg_basic.t_left)
np.save('t_matrix_fourier_right',alg_basic.t_right)


# In[65]:


np.save('background',background_im_basic)

