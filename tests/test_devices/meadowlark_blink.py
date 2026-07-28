from openwfs.devices import SLMBlinkHDMI
import numpy as np

slm = SLMBlinkHDMI(
    blink_path=r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\Blink_C_wrapper.dll",
    monitor_id=2,
    is_10bit=True,
    coordinate_system="full",
    load_lookutp_table=False,
    lookup_table = np.arange(1024),
)

lut = slm.linear_lookup_table()
slm.lookup_table = lut


slm.set_phases(2*np.pi -0.005)
slm.pixels.read()

del slm