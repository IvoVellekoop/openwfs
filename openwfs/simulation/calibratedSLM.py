from openwfs.devices.slm import SLM
import numpy as np
from openwfs.devices.slm.slm import FrameBufferReader
from openwfs.simulation.slm import PhaseToField
from openwfs.utilities.patterns import coordinate_range
from openwfs.core import Detector, Processor
from typing import Callable
from openwfs.utilities.utilities import set_extent
from private_openwfs.devices.microscope_future import WFSSettings, MicroscopeOfTheFuture


class CalibratedSLM(SLM):
    """
    A callibrated SLM that matches the physics properties of the real SLM.

    Args:
        filter: A callable that takes a field and returns the filtered field.
        physical_size: The physical size of the SLM using astropy units (width, height).
    """

    def __init__(
        self,
        physical_size: tuple,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not np.isscalar(self.amplitude) and self.amplitude.shape != self.shape:
            raise ValueError("amplitude must have the same shape as the SLM shape.")

        self.physical_size = physical_size