import set_path
from openwfs.devices import ScanningMicroscope
from openwfs.slm import SLM
from openwfs.algorithms import BasicFDR
import astropy.units as u
import numpy as np
from openwfs.slm.geometry import fill_transform
from openwfs.processors import SelectRoi

# running this without an available DAQ? the NI MAX application allows you to simulate a DAQ.
scanner = ScanningMicroscope(bidirectional=True, sample_rate=0.5 * u.MHz,
                             axis0=('Dev1/ao0', -1.0 * u.V, 1.0 * u.V),
                             axis1=('Dev1/ao1', -1.0 * u.V, 1.0 * u.V),
                             input=('Dev1/ai0', -1.0 * u.V, 1.0 * u.V),
                             data_shape=(256, 256),
                             scale=440 * u.um / u.V,
                             delay=1.0 * u.us,
                             padding=0.05)

slm = SLM(0, wavelength=804 * u.nm)

# hardcode offset, because our calibrations don't work yet
transform_matrix = np.array(fill_transform(slm, fit='short'))
transform_matrix = transform_matrix * 0.8  # scaling according to last
# from the old hardcoded offset, visually adjusted to be right
transform_matrix[2, :] = [-0.0147 / (0.4 + 0.5), 0.0036 / 0.5, 1]
slm.transform = transform_matrix

# again copied from earlier hardcodes
slm.lut_generator = lambda λ: np.arange(0, 0.2623 * λ.to(u.nm).value - 23.33) / 255
slm.wavelength = 804 * u.nm
im = scanner.read()
fdbk = SelectRoi(source=scanner)
wfs = BasicFDR(slm=slm, feedback=fdbk)

devices = {
    'cam': scanner,
    'slm': slm,
    'wfs': wfs}
