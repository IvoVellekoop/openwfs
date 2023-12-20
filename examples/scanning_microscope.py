import set_path
from openwfs.devices import ScanningMicroscope
from openwfs.slm import SLM
from openwfs.algorithms import FourierDualReference
import astropy.units as u
import numpy as np
from openwfs.slm.geometry import fill_transform
from openwfs.processors import SelectRoi

scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz,
                             axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                             axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V),
                             input=('Dev1/ai0', -1.0 * u.V, 1.0 * u.V),
                             data_shape=(256, 256),
                             scale=440 * u.um / u.V,
                             delay=1.0 * u.us,
                             padding=0.05)

devices = {'microscope': scanner}
