from .fourier import FourierDualRef
import numpy as np
from ..core import DataSource, PhaseSLM
from .utilities import analyze_phase_stepping
import os
import pickle

def get_neighbors(n, m):
    """Get the neighbors of a point in a 2D grid.

    Args:
        n (int): x-coordinate of the point.
        m (int): y-coordinate of the point.

    Returns:
        numpy.ndarray: Array containing the coordinates of the neighbors.
    """
    directions = [
        (-1, -1), (-1, 0),  (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),   (1, 0),   (1, 1)
    ]

    neighbors = [(n + dx, m + dy) for dx, dy in directions]
    return np.array(neighbors)


class CharacterisingFDR(FourierDualRef):
    """FourierDualRef algorithm with additional features for characterizing modes.

    Attributes:
        max_modes (int): The maximum number of modes to characterize.
        high_modes (int): The number of high modes to measure.
        high_phase_steps (int): The number of phase steps for high mode measurements.
        t_left (numpy.ndarray): Transmission values for the left side.
        t_right (numpy.ndarray): Transmission values for the right side.
        k_left (numpy.ndarray): Left k-space matrix.
        k_right (numpy.ndarray): Right k-space matrix.
        t_slm (numpy.ndarray): SLM transmission matrix.
        intermediates (bool): Flag to record intermediate enhancements.
        intermediate_enhancements (list): List to store intermediate enhancements.
        added_modes (list): List to store added modes information.

    Methods:
        execute(): Execute the characterizing FourierDualRef algorithm.
        measure_high_modes(t_fourier, kx_total, ky_total, side): Measure the high modes.
        record_intermediate_enhancement(t_fourier, kx_total, ky_total, side): Record intermediate enhancements.
        save_experiment(filename="experimental_data", directory=None): Save experimental data to a file.
    """

    def __init__(self,feedback: DataSource, slm: PhaseSLM, slm_shape = (500,500), phase_steps=4, overlap=0.1, max_modes=20, high_modes=0, high_phase_steps=16, intermediates=False):
        """

        Args:
            phase_steps (int): The number of phase steps.
            overlap (float): The overlap value.
            max_modes (int): The maximum number of modes to characterize.
            high_modes (int): The number of high modes to measure.
            high_phase_steps (int): The number of phase steps for high mode measurements.
            intermediates (bool): Flag to enable recording intermediate enhancements.
        """
        super().__init__(feedback, slm, slm_shape,None, None, phase_steps=phase_steps, overlap=overlap)
        self.max_modes = max_modes
        self.high_modes = high_modes
        self.high_phase_steps = high_phase_steps
        self.t_left = None
        self.t_right = None
        self.k_left = None
        self.k_right = None
        self.t_slm = None
        self.intermediates = intermediates
        if self.intermediates:
            self.intermediate_enhancements = []
            self.intermediate_t = []
        self.added_modes = [['Uncorrected enhancement']]

    def execute(self):
        """Execute the algorithm.

        Returns:
            numpy.ndarray: Final calculated transmission matrix for the SLM.
        """
        kx_total = []
        ky_total = []

        # measure the flat_wf value
        if self.intermediates:
            self.record_intermediate_enhancement([0], [0], [0], 0)

        for side in range(2):
            n = 0
            # we begin in K[0,0]
            self.k_x = np.zeros((1, 1), dtype=int)
            self.k_y = np.zeros((1, 1), dtype=int)
            t_fourier = np.array([])
            kx_total = np.array([])
            ky_total = np.array([])
            centers = np.array([[], []], dtype=int)

            while n < self.max_modes:

                measurements = self.single_side_experiment(np.vstack((self.k_x, self.k_y)),side)
                t_fourier = np.append(t_fourier, analyze_phase_stepping(measurements,axis=1).field)
                kx_total = np.append(kx_total, self.k_x)
                ky_total = np.append(ky_total, self.k_y)

                found_next_center = False
                ind = 1
                while found_next_center is False:
                    nth_highest_index = np.argsort(abs(t_fourier))[-ind]
                    in_x = np.isin(centers[0, :], kx_total[nth_highest_index])
                    in_y = np.isin(centers[1, :], ky_total[nth_highest_index])

                    if not any(in_x & in_y) is True:
                        center = np.array([[kx_total[nth_highest_index]], [ky_total[nth_highest_index]]])
                        next_points = get_neighbors(center[0], center[1])

                        # cull all the previously measured points
                        unique_next_points = np.array([point for point in next_points[:, :, 0] if not any(
                            (np.array(np.vstack((kx_total, ky_total))).T == point).all(1))])
                        centers = np.concatenate((centers, center), axis=1)

                        if len(unique_next_points) > 0:
                            found_next_center = True

                    ind += 1

                self.k_x = unique_next_points[:, 0]
                self.k_y = unique_next_points[:, 1]

                if (n + len(self.k_x)) < self.max_modes:
                    self.added_modes.append(unique_next_points.tolist())

                if n > 0 and self.intermediates:
                    self.record_intermediate_enhancement(t_fourier, kx_total, ky_total, side)

                n += len(self.k_x)

            # Measuring highest modes
            self.measure_high_modes(t_fourier, kx_total, ky_total, side)

            if self.high_modes != 0 and self.intermediates:
                self.record_intermediate_enhancement(t_fourier, kx_total, ky_total, side)

            if side == 0:
                self.t_left = t_fourier
                self.k_left = np.vstack((kx_total, ky_total))

            else:
                self.t_right = t_fourier
                self.k_right = np.vstack((kx_total, ky_total))

        self.t_slm = self.compute_t(self.t_left, self.t_right, self.k_left, self.k_right)
        return self.t_slm

    def measure_high_modes(self, t_fourier, kx_total, ky_total, side):
        """Measure the high-order modes of the system.

        Args:
            t_fourier (numpy.ndarray): Existing transmission coefficients.
            kx_total (numpy.ndarray): Total kx angles that have been measured.
            ky_total (numpy.ndarray): Total ky angles that have been measured.
            side (int): Which side (left or right) the measurements are taken from.
        """

        # Get indices of the n highest modes
        if self.high_modes == 0:
            return

        high_mode_indices = np.argsort(np.abs(t_fourier))[-self.high_modes:]

        # Store the original phase_steps
        original_phase_steps = self.phase_steps

        # Update phase_steps for high mode measurements
        self.phase_steps = self.high_phase_steps

        # Remeasure the highest modes
        for idx in high_mode_indices:
            self.k_x = np.array([kx_total[idx]])
            self.k_y = np.array([ky_total[idx]])
            measurements = self.single_side_experiment(np.vstack((self.k_x, self.k_y)),side)
            t_fourier[idx] = analyze_phase_stepping(measurements,axis=1).field

        # Restore the original phase_steps for subsequent measurements
        self.phase_steps = original_phase_steps
        self.added_modes.append([f'Remeasuring {self.high_modes} highest modes'])

    def record_intermediate_enhancement(self, t_fourier, kx_total, ky_total, side):
        """Record intermediate enhancement data for later analysis.

        Args:
            t_fourier (numpy.ndarray): Existing transmission coefficients.
            kx_total (numpy.ndarray): Total kx angles that have been measured.
            ky_total (numpy.ndarray): Total ky angles that have been measured.
            side (int): Which side (left or right) the measurements are taken from.
        """
        k = np.vstack((kx_total, ky_total))

        if side == 0:
            t_left = t_fourier
            k_left = k
            # Use the class-level attributes for the right side, if they're set; otherwise, initialize to zeros
            t_right = np.zeros_like(t_fourier)
            k_right = k

        else:
            t_right = t_fourier
            k_right = k

            t_left = self.t_left  # Since side == 0 is always processed first, self.t_left should always be set by this point
            k_left = self.k_left

        t_slm = self.compute_t(t_left, t_right, k_left, k_right)
        self.intermediate_t.append(t_slm)
        self._slm.set_phases(np.angle(t_slm))

        self.intermediate_enhancements.append(self._feedback.read())

    def save_experiment(self, filename="experimental_data", directory=None):
        """Save the experimental data to a specified directory.

        Args:
            filename (str): Name of the file to save the data.
            directory (str): Directory to save the file in. If None, uses the current directory.
        """

        if directory is None:
            directory = os.getcwd()  # Get current directory

        data_to_save = {
            "t_left": self.t_left,
            "t_right": self.t_right,
            "k_left": self.k_left,
            "k_right": self.k_right,
            "t_slm": self.t_slm,
            "intermediate_enhancement": self.intermediate_enhancements,
            "added_modes": self.added_modes
        }

        with open(os.path.join(directory, f'{filename}.pkl'), 'wb') as f:
            pickle.dump(data_to_save, f)
