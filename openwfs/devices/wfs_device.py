import numpy as np


class WFSController:
    def __init__(self, algorithm):
        """
        Initializes the FDRController with an algorithm ojbect.

        Args:
            algorithm: An instance of an algorithm class. (e.g. StepwiseSequential, BasicFDR, CharacterisingFDR)
        """
        self.algorithm = algorithm
        self._show_flat_wavefront = False
        self._show_optimized_wavefront = False
        self.transmission_matrix = None

    @property
    def execute_button(self) -> bool:
        """
        Property to trigger the execution of the BasicFDR instance's method.

        Returns:
            bool: The state of the execution.
        """
        return self.algorithm.execute_button

    @execute_button.setter
    def execute_button(self, value):
        self.transmission_matrix = self.algorithm.execute()
        self.algorithm.execute_button = value

    @property
    def show_flat_wavefront(self) -> bool:
        """
        Property to show a flat wavefront.

        Returns:
            bool: The state of the flat wavefront display.
        """
        return self._show_flat_wavefront

    @show_flat_wavefront.setter
    def show_flat_wavefront(self, value):
        if value:
            self.algorithm._slm.set_phases(0)  # Assuming there's a method in BasicFDR to set all phases to 0
        self._show_flat_wavefront = value

    @property
    def show_optimized_wavefront(self) -> bool:
        """
        Property to show the optimized wavefront.

        Returns:
            bool: The state of the optimized wavefront display.
        """
        return self._show_optimized_wavefront

    @show_optimized_wavefront.setter
    def show_optimized_wavefront(self, value):
        if value:
            if len(self.transmission_matrix.shape) == 3:
                self.algorithm._slm.set_phases(-np.angle(self.transmission_matrix[...,0]))
            else:
                self.algorithm._slm.set_phases(np.angle(self.transmission_matrix))
        self._show_optimized_wavefront = value