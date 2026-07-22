from openwfs.devices import SLMBlinkHDMI
import numpy as np

slm = SLMBlinkHDMI(blink_path=r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK\Blink_C_wrapper.dll", monitor_id=2, is_10bit=True, coordinate_system = "full")

slm.load_linear_lookup_table()

slm.set_phases(np.random.rand(10,10))
slm.temperature

del slm