import set_path
from openwfs.devices import GalvoScanner,LaserScanning
from openwfs.slm import SLM
from openwfs.algorithms import BasicFDR
import astropy.units as u
import numpy as np
from openwfs.slm.geometry import fill_transform
from openwfs.processors import SelectRoi

# running this without an available DAQ? the NI MAX application allows you to simulate a DAQ.
g = GalvoScanner()
scanner = LaserScanning(dwelltime=6*u.ms,x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3', input_mapping='Dev4/ai24', galvo_scanner=g)
slm = SLM(0,wavelength=804*u.nm)
scanner.width = 60
scanner.height = 60
# hardcode offset, because our calibrations don't work yet
transform_matrix = np.array(fill_transform(slm, fit='short'))
transform_matrix = transform_matrix * 0.8  # scaling according to last
transform_matrix[2, :] = [-0.0147 / (0.4 + 0.5), 0.0036 / 0.5,
                          1]  # from the old hardcoded offset, visually adjusted to be right
slm.transform = transform_matrix

slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33)/255 # again copied from earlier hardcodes
slm.wavelength = 804 * u.nm


scanner.measurement_time = 3*u.ms
fdbk = SelectRoi(source=scanner)
wfs = BasicFDR(slm = slm, feedback=fdbk)

# wfs.execute()
devices = {
    'cam': scanner,
    'g': g,
    'slm': slm,
    'wfs': wfs}

