from .fourier import FourierBase
import numpy as np
from ..core import Detector, PhaseSLM
from .utilities import analyze_phase_stepping, WFSResult
import os
import pickle
from typing import List


def get_neighbors(n, m):
    """Get the neighbors of a point in a 2D grid.

    Args:
        n (int): x-coordinate of the point.
        m (int): y-coordinate of the point.

    Returns:
        numpy.ndarray: Array containing the coordinates of the neighbors.
    """
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    neighbors = [(n + dx, m + dy) for dx, dy in directions]
    return np.array(neighbors)


class PathfindingFourier(FourierBase):
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

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape=(500, 500), phase_steps=4, overlap=0.1,
                 max_modes=20, high_modes=0, high_phase_steps=16, intermediates=False):
        """

        Args:
            feedback (Detector): Source of feedback
            slm (PhaseSLM): The spatial light modulator
            slm_shape (tuple of two ints): The shape that the SLM patterns & transmission matrices are calculated for,
                                        does not necessarily have to be the actual pixel dimensions as the SLM.
            phase_steps (int): The number of phase steps.
            overlap (float): The overlap value.
            max_modes (int): The maximum number of modes to characterize.
            high_modes (int): The number of high modes to measure.
            high_phase_steps (int): The number of phase steps for high mode measurements.
            intermediates (bool): Flag to enable recording intermediate enhancements.
        """
        super().__init__(feedback, slm, slm_shape, None, None, phase_steps=phase_steps, overlap=overlap)
        self.max_modes = max_modes
        self.high_modes = high_modes
        self.high_phase_steps = high_phase_steps
        self.t_left = None
        self.t_right = None
        self.k_x = None
        self.k_y = None
        self.t_slm = None

        self.feedback_target = []  # Pathfinding only works for 1-dimensional feedback.

    def execute(self) -> WFSResult:
        """Execute the algorithm.

        Returns:
            WFSResult: Final calculated transmission matrix for the SLM.
        """
        left_results: List[WFSResult] = []
        right_results: List[WFSResult] = []

        for side in range(2):
            n: int = 0
            self.k_x = self.k_y = np.array(0, dtype=int)
            kx_total: np.ndarray = np.array([])
            ky_total: np.ndarray = np.array([])
            centers: np.ndarray = np.array([[], []], dtype=int)

            while n < self.max_modes:
                wfs_result: WFSResult = self.single_side_experiment(np.vstack((self.k_x, self.k_y)), side)
                if side == 0:
                    left_results.append(wfs_result)
                else:
                    right_results.append(wfs_result)

                kx_total = np.append(kx_total, self.k_x)
                ky_total = np.append(ky_total, self.k_y)

                # Combine results up to the current point
                combined_result: WFSResult = self.combine_results(left_results if side == 0 else right_results)
                # Find next center for measurements
                found_next_center: bool = False
                ind: int = 1
                while not found_next_center:
                    nth_highest_index: int = np.argsort(np.abs(combined_result.t))[-ind]
                    in_x = np.isin(centers[0, :], kx_total[nth_highest_index])
                    in_y = np.isin(centers[1, :], ky_total[nth_highest_index])

                    if not any(in_x & in_y):
                        center = np.array([[kx_total[nth_highest_index]], [ky_total[nth_highest_index]]])
                        next_points: np.ndarray = get_neighbors(center[0, 0], center[1, 0])
                        unique_next_points: List[np.ndarray] = [
                            point for point in next_points if
                            tuple(point) not in set(zip(kx_total, ky_total))
                        ]
                        centers = np.concatenate((centers, center), axis=1)

                        if unique_next_points:
                            found_next_center = True
                    ind += 1

                if unique_next_points:
                    self.k_x = np.array(unique_next_points)[:, 0]
                    self.k_y = np.array(unique_next_points)[:, 1]
                else:
                    break  # Exit if no new points are found

                n += len(self.k_x)

            if side == 0:
                self.k_left = np.vstack((kx_total, ky_total))
            else:
                self.k_right = np.vstack((kx_total, ky_total))

        self.results_left = self.combine_results(left_results)
        self.results_right = self.combine_results(right_results)

        results_slm = self._compute_t(self.results_left, self.results_right, self.k_left, self.k_right)
        return results_slm

    def combine_results(self, results: List[WFSResult]) -> WFSResult:
        """Combine multiple WFSResult objects into a single WFSResult.

        Args:
            results (List[WFSResult]): List of WFSResult objects to combine.

        Returns:
            WFSResult: Combined result.

        ToDo: i am not sure if the snr properties get passed through well. look into this.
        """
        t_combined = np.concatenate([result.t for result in results], axis=0)
        snr_combined = np.concatenate([result.snr for result in results])
        amplitude_factor_combined = np.concatenate([result.amplitude_factor for result in results])
        estimated_improvement_combined = np.concatenate([result.estimated_improvement for result in results])
        n_combined = np.sum([result.n for result in results])

        return WFSResult(t=t_combined,
                         snr=snr_combined,
                         amplitude_factor=amplitude_factor_combined,
                         estimated_improvement=estimated_improvement_combined,
                         n=n_combined)

    @property
    def execute_button(self) -> bool:
        return self._execute_button

    @execute_button.setter
    def execute_button(self, value):
        self.execute()
        self._execute_button = value
