from openwfs.devices.slm import SLM
import numpy as np
from openwfs.devices.slm.slm import FrameBufferReader
from openwfs.simulation.slm import PhaseToField
from openwfs.utilities.patterns import coordinate_range
from openwfs.core import Detector, Processor
from typing import Callable

class CalibratedSLM(SLM):
    """
    A callibrated SLM that matches the physics properties of the real SLM.
    
    Args:
        filter: A callable that takes a field and returns the filtered field.
        physical_size: The physical size of the SLM using astropy units (width, height).
        wavelength: The wavelength of the light in meters.
    """
    def __init__(self, physical_size: tuple, wavelength: float, filter: Callable[[np.ndarray], np.ndarray] = None,*args, **kwargs):           
        super().__init__(*args, **kwargs)

        if not np.isscalar(self.amplitude) and self.amplitude.shape != self.shape:
            raise ValueError("amplitude must have the same shape as the SLM shape.")
        
        self.filter = filter
        self.physical_size = physical_size
        self.wavelength = wavelength


    # def pupil2kspace_coordinates(self):
    #     """
    #     Convert pupil coordinates to k-space coordinates.
    #     """
    #     # Assuming the SLM has a square shape
    #     slm_shape = self.shape
    #     transform = self.transform

    #     # convert pupil coordinates to k-space coordiants
    #     if self._coordiante_system == "full":
    #         slm_pupil_coords = coordinate_range(slm_shape, extent = 2)
    #         slm_pupil_coords_y = slm_pupil_coords[0]
    #         slm_pupil_coords_x = slm_pupil_coords[1]
    #         kx = None
    #         ky = None
    #     else: 
    #         raise ValueError("Only full coordinate system is supported for pupil to k-space conversion.")

    #         # convert pupil coordinates to k-space coordinates
    #     return kx, ky    

    
class FilterPropagate(Processor):
    """
    Take a phase as input, build the field and filter it with the SLM filter.
    Computes 'Filter[moduled_field_amplitude * exp(1j * phase) + non_modulated_field_fraction]'.
    """

    def __init__(self, incident_field: CalibratedSLM, distance, diameter):
        self.distance = distance
        self.diameter = diameter
        self._incident_field = incident_field
        super().__init__(incident_field)

    def _fetch(self, incident_field: np.ndarray) -> np.ndarray:
        """
        Compute the field at the SLM plane given the SLM phases.
        """
        # compute coordinates from source SLM
        x, y = self._incident_field._coordinates()
        # propagation...

        return self.slm_filter()