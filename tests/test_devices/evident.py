import private_openwfs.devices as powfs_d
from openwfs.utilities import get_pixel_size
import astropy.units as u
import time
import numpy as np
import matplotlib.pyplot as plt

url = "http://127.0.0.1:8080/xmlrpc"

mic = powfs_d.EvidentMicroscope(url)
img = mic.cam.read()  # start imaging
mic.cam.trigger(out=img)
# mic.cam.wait() # Error on this line but this should work. Problem with how evident is mapped into the openwfs interface
img = mic.cam.read()
mic.cam.stop_imaging()

pixel_size = get_pixel_size(img)
assert pixel_size is not None
assert len(pixel_size) == 2

channels_id = mic.get_channel_list()

try:
    mic.get_channel_enabled(channel_id=channels_id[0])
    mic.get_channel_hv(channel_id=channels_id[0])
    mic.get_channel_gain(channel_id=channels_id[0])
    mic.get_channel_offset(channel_id=channels_id[0])
except Exception as e:
    print("Error getting channel settings:", e)

lasers_id = mic.get_laser_list_settings()
mic.get_channel_laser(channels_id[0])
mic.get_laser_enabled(lasers_id[0]["laserId"])
mic.get_laser_intensity(lasers_id[0]["laserId"])

mic.objective
mic.objective_list
mic.z_stack_slice_num
mic.z_stack_step_size
x, y = mic.stage_xy.xy  # get current stage position

mic.stage_xy.xy = (3 * u.cm, 3 * u.cm)  # move stage to origin
# data1 = mic.cam.read()
mic.stage_xy.xy = (0 * u.nm, 0 * u.nm)  # move stage to (0,0) in nm
mic.stage_xy.xy = (x, y)  # move stage to (x,y) in nm

mic.objective_list

mic.objective = "UPLXAPO4X"  # set objective to 0, magnification 4x
mic.objective  # get current objective lens name
mic.z_stack_enabled = False
mic.z_stack_enabled
mic.t_stack_enabled = False
mic.t_stack_enabled

img = mic.cam.read()  # start imaging
# print size of img in pixels
print("Image shape (pixels):", img.shape)

z_vec = np.arange(5) * u.um + 1000 * u.um
mic.z_stack_read(z_vec)

t = np.arange(5) * u.s
mic.t_stack_read(t)
