import set_path
from openwfs.devices import LaserScanning, Gain
import astropy.units as u

scanner = LaserScanning(
    x_mirror_mapping='Dev4/ao2',
    y_mirror_mapping='Dev4/ao3',
    input_mapping='Dev4/ai24',
    data_shape=(500, 500),
    duration=1000 * u.ms,
)

gain = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0",
    reset=False,
    gain=0,
)

devices = {'cam': scanner, 'gain': gain}
