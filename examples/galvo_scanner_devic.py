import set_path
from openwfs.devices import LaserScanning, Gain


scanner = LaserScanning(x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3', input_mapping='Dev4/ai24')
gain = Gain(gain=0)

devices = {'scanner': scanner, 'gain': gain}

