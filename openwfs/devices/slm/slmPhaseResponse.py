from . import SLM

from ...core import Detector
from ...processors import LookupTableProcessor, ModProcessor
from typing import Tuple
import numpy as np


class SLMPhaseResponse(SLM):
    """
    A subclass of SLM that represents a phase-only spatial light modulator (SLM) with a specific phase response.
    """

    def __init__(
        self,
        phase_response: Tuple,
        method: str = "nearest",
        fill_value: None | float = None,
        bounds_error: None | bool = None,
        solver: None = None,
        **kwargs,
    ):
        """
        Initializes the SLMPhaseResponse object. This is the SLM object that should be used for simulation. Relatively to the SLM object, the SLMPhaseResponse allows to simulate the effect of both lookup tables and phase response (i.e. how phases are mapped into bit value and from bit value to phase value experience by light)

        Args:
            phase_response: A tuple specifying the bit value (screen pixel color) is converted to phase. The phase response maps the np.array of phases_response[0] to the np.array of phases_response[1].
            method: The interpolation method to use. Defaults to nearest interpolation See :py:class:`~openwfs.processors.LookupTableProcessor` for details.
            fill_value: The value to use for points outside of the interpolation range. See :py:class:`~openwfs.processors.LookupTableProcessor` for details.
            bounds_error: Whether to raise an error when interpolating outside of the bounds. See :py:class:`~openwfs.processors.LookupTableProcessor` for details.
            solver: The solver to use for the interpolation. See :py:class:`~openwfs.processors.LookupTableProcessor` for details.
            **kwargs: all the arguments that are passed to the SLM class. See :py:class:`~openwfs.devices.slm.SLM` for details.
        """
        self._phase_response = phase_response
        super().__init__(**kwargs)
        self.itp_args = {
            "method": method,
            "fill_value": fill_value,
            "bounds_error": bounds_error,
            "solver": solver,
        }
        self.lookup_table = None

        self.update_processor()

    @property
    def phases(self) -> LookupTableProcessor:
        """
        Returns the default phases of the SLM with a processor that applies the phase response to the default phases.
        """
        return self._lut_processor

    @property
    def phase_response(self) -> Tuple:
        """
        Returns the phase response of the SLM.
        """
        return self._phase_response

    @phase_response.setter
    def phase_response(self, value: Tuple) -> None:
        self._phase_response = value
        self.update_processor()

    def update_processor(self) -> None:
        """
        Returns the processor that applies the phase response to the default phases.
        """
        self._mod_processor = ModProcessor(super().phases, 2 * np.pi)

        # Applies mapping from phase to grey value
        angle = np.linspace(0, 2 * np.pi, self.lookup_table.size + 1, endpoint=True)
        lookup_table_to_screen = (
            angle + np.diff(angle[0:2])[0] / 4,
            np.append(self.lookup_table, self.lookup_table[0]),
        )
        self._bit_on_slm = LookupTableProcessor(self._mod_processor, lookup_table_to_screen, method="nearest")

        # Applies mapping from grey value to phase value
        self._lut_processor = LookupTableProcessor(self._bit_on_slm, self._phase_response, **(self.itp_args))

    @property
    def lookup_table(self):
        return super().lookup_table

    @lookup_table.setter
    def lookup_table(self, lut):
        SLM.lookup_table.fset(self, lut)
        self.update_processor()
