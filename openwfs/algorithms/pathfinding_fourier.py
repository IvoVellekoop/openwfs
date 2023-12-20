from .fourier import FourierBase
import numpy as np
from ..core import Detector, PhaseSLM
from .utilities import analyze_phase_stepping, WFSResult
import os
import pickle
from typing import List


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

        for side in range(2):
            results = self.single_side_pathfinding(side)

            if side == 0:
                self.results_left = self.combine_results(results)
            else:
                self.results_right = self.combine_results(results)

        results_slm = self._compute_t(self.results_left, self.results_right, self.k_left, self.k_right)
        return results_slm

    def single_side_pathfinding(self, side: int) -> List[WFSResult]:
        """
        Performs the pathfinding experiment for a single side. It works by performing small (1 to 8 modes at a time)
        experiments and choosing the next modes by finding the highest current modes.

        The algorithm works as such: First it measures the modes it has in its self.k_x and self.k_y parameters.
        Then, the algorithm looks at the whole set of measurements including earlier ones, and finds the highest
        measured mode that has available unmeasured modes directly around it.

        Then, it selects these modes as the next measurement. This is iteratively repeated until the algorithm would
        be going over the maximum number of modes. If that's the case, the next modes are not measured, and the list
        of WFSResults is returned.

        Args:
            side (bool): 0 for left, 1 for the right SLM pupil

        Returns: a list of WFSResults. One for each step in the pathfinding algorithm

        """
        results = []
        centers = np.array([[], []], dtype=int)
        kx_total = np.array([])
        ky_total = np.array([])
        self.k_x = self.k_y = np.array(0, dtype=int)
        current_mode_count = 0

        while current_mode_count < self.max_modes:
            wfs_result = self.single_side_experiment(np.vstack((self.k_x, self.k_y)), side)     # do the measurements

            results.append(wfs_result)
            kx_total, ky_total = np.append(kx_total, self.k_x), np.append(ky_total, self.k_y)

            combined_result = self.combine_results(results) # combine all previous results to find the new center
            unique_next_points = self.find_next_center(combined_result, centers, kx_total, ky_total)

            self.k_x, self.k_y = np.array(unique_next_points)[:, 0], np.array(unique_next_points)[:, 1]
            current_mode_count += len(self.k_x)

        if side == 0:
            self.k_left = np.vstack((kx_total, ky_total))
        else:
            self.k_right = np.vstack((kx_total, ky_total))

        return results

    def find_next_center(self, combined_result: WFSResult, centers: np.ndarray, kx_total: np.ndarray,
                          ky_total: np.ndarray) -> List[np.ndarray]:
        """
        Looks at the currently measured modes and selects the highest mode that has measurable modes around it.
        Because previously measured modes never have measurable modes around it, we keep track of them as well.
        This is unnecessary, but a bit more efficient.

        Args:
            combined_result (WFSResult): combined WFSResult object of all previously measured modes.
            centers (np.ndarray): Array containing the previously measured centers.
            kx_total (np.ndarray): Array containing the k-space X coordinates in order.
            ky_total (np.ndarray): Array containing the k-space Y coordinates in order.

        Returns:

        """
        found_next_center = False
        ind = 1
        unique_next_points = []

        while not found_next_center:
            nth_highest_index = np.argsort(np.abs(combined_result.t))[-ind]
            in_x = np.isin(centers[0, :], kx_total[nth_highest_index])
            in_y = np.isin(centers[1, :], ky_total[nth_highest_index])

            if not any(in_x & in_y):
                center = np.array([[kx_total[nth_highest_index]], [ky_total[nth_highest_index]]])
                next_points = self.get_neighbors(center[0, 0], center[1, 0])
                unique_next_points = [point for point in next_points if
                                      tuple(point) not in set(zip(kx_total, ky_total))]
                centers = np.concatenate((centers, center), axis=1)

                if unique_next_points:
                    found_next_center = True
            ind += 1

        return unique_next_points

    def combine_results(self, results: List[WFSResult]) -> WFSResult:
        """Combine multiple WFSResult objects into a single WFSResult.

        Args:
            results (List[WFSResult]): List of WFSResult objects to combine.

        Returns:
            WFSResult: Combined result.

        ToDo: I really don't like averaging them. It's great to concatenate them! that would be really nice to see the
            results step-by-step. Unfortunately we have do it because _compute_t assumes k-space symmetry.
        """
        t_combined = np.concatenate([result.t for result in results], axis=0)
        snr_combined = np.mean([result.snr for result in results])
        amplitude_factor_combined = np.mean([result.amplitude_factor for result in results])
        estimated_improvement_combined = np.mean([result.estimated_improvement for result in results])

        # snr_combined = np.concatenate([result.snr for result in results])
        # amplitude_factor_combined = np.concatenate([result.amplitude_factor for result in results])
        # estimated_improvement_combined = np.concatenate([result.estimated_improvement for result in results])
        # n_combined = np.sum([result.n for result in results])

        return WFSResult(t=t_combined,
                         snr=snr_combined,
                         amplitude_factor=amplitude_factor_combined,
                         estimated_improvement=estimated_improvement_combined,
                         n=None)

    def get_neighbors(self, n, m):
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

    @property
    def execute_button(self) -> bool:
        return self._execute_button

    @execute_button.setter
    def execute_button(self, value):
        self.execute()
        self._execute_button = value
