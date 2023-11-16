import pytest
from ..openwfs.devices import GalvoScanner,LaserScanning
import time
import astropy.units as u

def test_measurement_time():
    # This tests if the measurement time actually changes the time the scanners measures. Requires connection to DAQ
    # or a simulated DAQ in NI MAX
    g = GalvoScanner()
    scanner = LaserScanning(dwelltime=60 * u.ms, x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3',
                            input_mapping='Dev4/ai24', galvo_scanner=g)
    diffs = []
    for times in range(6):

        scanner.measurement_time = times/10 * u.s
        t = time.time()
        scanner.trigger()
        scanner.wait()
        scanner.read()
        diffs.append((time.time() - t)-(times/10))

    for diff in diffs:
        assert diff < 0.01