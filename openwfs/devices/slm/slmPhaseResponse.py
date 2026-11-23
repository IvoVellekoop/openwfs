from . import SLM

from ...core import Detector
from ...processors import LookupTableProcessor, ModProcessor
from typing import Tuple
import numpy as np


class SLMPhaseResponse(SLM):
    """
    A subclass of SLM that represents a phase-only spatial light modulator (SLM) with a specific phase response.

    This class allows for the simulation of an SLM that modulates the phase of an incident light field based on a given phase response function. The phase response can be defined by the user and is applied to the incident field to produce a modulated output field.

    Attributes:
        phase_response (callable): A function that defines the phase response of the SLM. It should take an array of input values and return an array of phase values.
    """

    def __init__(self, phase_response: Tuple, **kwargs):
        """
        Initializes the SLMPhaseResponse object.

        Args:
            phase_response: A tuple specifying a lookup table. The phase response maps the np.array of phases_response[0] to the np.array of phases_response[1]. The mapping is achieving using linear interpolation.
            kwargs: all the arguments that are passed to the SLM class. See :py:class:`~openwfs.devices.slm.SLM` for details.
        """
        self._phase_response = phase_response
        super().__init__(**kwargs)

    @property
    def phases(self) -> Detector:
        """
        Returns the default phases of the SLM with a processor that applies the phase response to the default phases.
        """
        default_phases = super().phases
        mod_phases = ModProcessor(default_phases, 2 * np.pi)
        return LookupTableProcessor(mod_phases, self._phase_response)
