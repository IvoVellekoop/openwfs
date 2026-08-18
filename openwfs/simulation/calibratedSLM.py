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

class FilterPropagate(Processor):
    """
    Computes 'Filter[slm.field]'.
    """

    def __init__(self, incident_field: CalibratedSLM, Microscope: MicroscopeOfTheFuture):
        self.microscope = Microscope
        self._incident_field = incident_field
        super().__init__(incident_field)

    def _fetch(self, incident_field: np.ndarray) -> np.ndarray:
        """
        Compute the field at the SLM plane given the SLM phases.
        """
        incident_field = set_extent(incident_field, self.microscope.wfs_module.settings.slm_physical_size)
        incident_field = MicroscopeOfTheFuture.filter_hole(incident_field, self.microscope.wavelength, self.microscope.wfs_module.settings.distance_mic_galvo, self.microscope.wfs_module.settings.d_beam, self.microscope.wfs_module.settings.d_mic_tube)
        return self.slm_filter()
