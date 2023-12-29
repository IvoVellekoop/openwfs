import set_path  # noqa - needed for setting the module search path to find openwfs
from openwfs.devices import ScanningMicroscope, Gain
from openwfs.slm import SLM
from openwfs.algorithms import FourierDualReference
import astropy.units as u
import numpy as np
from openwfs.slm.geometry import fill_transform
from openwfs.processors import SelectRoi
from nidaqmx.constants import TerminalConfiguration
"""
Example containing the creation of a ScanningMicroscope and Gain object. This script is intended to be 
    read by MicroManager's PyDevice.
"""

max_FOV_V = 1.0 * u.V

scanner = ScanningMicroscope(
    bidirectional=True,
    sample_rate=0.2 * u.MHz,
    axis0=('Dev4/ao2', -max_FOV_V, max_FOV_V),
    axis1=('Dev4/ao3', -max_FOV_V, max_FOV_V),
    input=('Dev4/ai16', -1.0 * u.V, 1.0 * u.V, TerminalConfiguration.DIFF),
    data_shape=(300, 300),
    scale=440 * u.um / u.V,
    delay=58.0,
    padding=0.05)

gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0",
    reset=False,
    gain=0.65 * u.V,
)

devices = {
    'scanner': scanner,
    'gain': gain,
}
