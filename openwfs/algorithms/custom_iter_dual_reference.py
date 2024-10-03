from typing import Optional

import numpy as np
from numpy import ndarray as nd

from .utilities import analyze_phase_stepping, WFSResult
from ..core import Detector, PhaseSLM


def weighted_average(a, b, wa, wb):
    """
    Compute the weighted average of two values.

    Args:
        a: The first value.
        b: The second value.
        wa: The weight of the first value.
        wb: The weight of the second value.
    """
    return (a * wa + b * wb) / (wa + wb)


class IterativeDualReference:
    """
    A generic iterative dual reference WFS algorithm, which can use a custom set of basis functions.

    This algorithm is adapted from [1], with the addition of the ability to use custom basis functions and specify the number of iterations.

    In this algorithm, the SLM pixels are divided into two groups: A and B, as indicated by the boolean group_mask argument.
    The algorithm first keeps the pixels in group B fixed, and displays a sequence on patterns on the pixels of group A.
    It uses these measurements to construct an optimized wavefront that is displayed on the pixels of group A.
    This process is then repeated for the pixels of group B, now using the *optimized* wavefront on group A as reference.
    Optionally, the process can be repeated for a number of iterations, which each iteration using the current correction
     pattern as a reference. This makes this algorithm suitable for non-linear feedback, such as multi-photon
    excitation fluorescence [2].

    This algorithm assumes a phase-only SLM. Hence, the input modes are defined by passing the corresponding phase
    patterns (in radians) as input argument.

    [1]: X. Tao, T. Lam, B. Zhu, et al., “Three-dimensional focusing through scattering media using conjugate adaptive
    optics with remote focusing (CAORF),” Opt. Express 25, 10368–10383 (2017).

    [2]: Gerwin Osnabrugge, Lyubov V. Amitonova, and Ivo M. Vellekoop. "Blind focusing through strongly scattering media
    using wavefront shaping with nonlinear feedback", Optics Express, 27(8):11673–11688, 2019.
    https://opg.optica.org/oe/ abstract.cfm?uri=oe-27-8-1167
    """

    def __init__(
        self,
        feedback: Detector,
        slm: PhaseSLM,
        phase_patterns: tuple[nd, nd],
        group_mask: nd,
        phase_steps: int = 4,
        iterations: int = 4,
        analyzer: Optional[callable] = analyze_phase_stepping,
    ):
        """
        Args:
            feedback: The feedback source, usually a detector that provides measurement data.
            slm: Spatial light modulator object.
            phase_patterns: A tuple of two 3D arrays, containing the phase patterns for group A and group B, respectively.
                The first two dimensions are the spatial dimensions, and should match the size of group_mask.
                The 3rd dimension in the array is index of the phase pattern. The number of phase patterns in A and B may be different.
            group_mask: A 2D bool array of that defines the pixels used by group A with False and elements used by
                group B with True.
            phase_steps: The number of phase steps for each mode (default is 4). Depending on the type of
                non-linear feedback and the SNR, more might be required.
            iterations: Number of times to measure a mode set, e.g. when iterations = 5, the measurements are
                A, B, A, B, A. Should be at least 2
            analyzer: The function used to analyze the phase stepping data. Must return a WFSResult object. Defaults to `analyze_phase_stepping`
        """
        if (phase_patterns[0].shape[0:2] != group_mask.shape) or (phase_patterns[1].shape[0:2] != group_mask.shape):
            raise ValueError("The phase patterns and group mask must all have the same shape.")
        if iterations < 2:
            raise ValueError("The number of iterations must be at least 2.")
        if np.prod(feedback.data_shape) != 1:
            raise ValueError("The feedback detector should return a single scalar value.")

        self.slm = slm
        self.feedback = feedback
        self.phase_steps = phase_steps
        self.iterations = iterations
        self.analyzer = analyzer
        self.phase_patterns = (phase_patterns[0].astype(np.float32), phase_patterns[1].astype(np.float32))
        mask = group_mask.astype(bool)
        self.masks = (~mask, mask)  # masks[0] is True for group A, mask[1] is True for group B

        # Pre-compute the conjugate modes for reconstruction
        self.modes = [
            np.exp(-1j * self.phase_patterns[side]) * np.expand_dims(self.masks[side], axis=2) for side in range(2)
        ]

    def execute(self, capture_intermediate_results: bool = False, progress_bar=None) -> WFSResult:
        """
        Executes the blind focusing dual reference algorithm and compute the SLM transmission matrix.
            capture_intermediate_results: When True, measures the feedback from the optimized wavefront after each iteration.
                This can be useful to determine how many iterations are needed to converge to an optimal pattern.
                This data is stored as the 'intermediate_results' field in the results
            progress_bar: Optional progress bar object. Following the convention for tqdm progress bars,
                this object should have a `total` attribute and an `update()` function.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data. The amplitude profile
                of each mode is assumed to be 1. If a different amplitude profile is desired, this can be obtained by
                multiplying that amplitude profile with this transmission matrix.
        """

        # Current estimate of the transmission matrix (start with all 0)
        t_full = np.zeros(shape=self.modes[0].shape[0:2])
        t_other_side = t_full

        # Initialize storage lists
        t_set_all = [None] * self.iterations
        results_all = [None] * self.iterations  # List to store all results
        results_latest = [None, None]  # The two latest results. Used for computing fidelity factors.
        intermediate_results = np.zeros(self.iterations)  # List to store feedback from full patterns

        # Prepare progress bar
        if progress_bar:
            num_measurements = (
                np.ceil(self.iterations / 2) * self.modes[0].shape[2]
                + np.floor(self.iterations / 2) * self.modes[1].shape[2]
            )
            progress_bar.total = num_measurements

        # Switch the phase sets back and forth multiple times
        for it in range(self.iterations):
            side = it % 2  # pick set A or B for phase stepping
            ref_phases = -np.angle(t_full)  # use the best estimate so far to construct an optimized reference
            side_mask = self.masks[side]
            # Perform WFS experiment on one side, keeping the other side sized at the ref_phases
            result = self._single_side_experiment(
                mod_phases=self.phase_patterns[side],
                ref_phases=ref_phases,
                mod_mask=side_mask,
                progress_bar=progress_bar,
            )

            # Compute transmission matrix for the current side and update
            # estimated transmission matrix
            t_this_side = self.compute_t_set(result, self.modes[side])
            t_full = t_this_side + t_other_side
            t_other_side = t_this_side

            # Store results
            t_set_all[it] = t_this_side  # Store transmission matrix
            results_all[it] = result  # Store result
            results_latest[side] = result  # Store latest result for this set

            # Try full pattern
            if capture_intermediate_results:
                self.slm.set_phases(-np.angle(t_full))
                intermediate_results[it] = self.feedback.read()

        # Compute average fidelity factors
        fidelity_noise = weighted_average(
            results_latest[0].fidelity_noise, results_latest[1].fidelity_noise, results_latest[0].n, results_latest[1].n
        )
        fidelity_amplitude = weighted_average(
            results_latest[0].fidelity_amplitude,
            results_latest[1].fidelity_amplitude,
            results_latest[0].n,
            results_latest[1].n,
        )
        fidelity_calibration = weighted_average(
            results_latest[0].fidelity_calibration,
            results_latest[1].fidelity_calibration,
            results_latest[0].n,
            results_latest[1].n,
        )

        result = WFSResult(
            t=t_full,
            t_f=None,
            n=self.modes[0].shape[2] + self.modes[1].shape[2],
            axis=2,
            fidelity_noise=fidelity_noise,
            fidelity_amplitude=fidelity_amplitude,
            fidelity_calibration=fidelity_calibration,
        )

        # TODO: document the t_set_all and results_all attributes
        result.t_set_all = t_set_all
        result.results_all = results_all
        result.intermediate_results = intermediate_results
        return result

    def _single_side_experiment(self, mod_phases: nd, ref_phases: nd, mod_mask: nd, progress_bar=None) -> WFSResult:
        """
        Conducts experiments on one part of the SLM.

        Args:
            mod_phases: 3D array containing the phase patterns of each mode. Axis 0 and 1 are used as spatial axis.
                Axis 2 is used for the 'phase pattern index' or 'mode index'.
            ref_phases: 2D array containing the reference phase pattern.
            mod_mask: 2D array containing a boolean mask, where True indicates the modulated part of the SLM.
            progress_bar: Optional progress bar object. Following the convention for tqdm progress bars,
                this object should have a `total` attribute and an `update()` function.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        num_modes = mod_phases.shape[2]
        measurements = np.zeros((num_modes, self.phase_steps))

        for m in range(num_modes):
            phases = ref_phases.copy()
            modulated = mod_phases[:, :, m]
            for p in range(self.phase_steps):
                phi = p * 2 * np.pi / self.phase_steps
                # set the modulated pixel values to the values corresponding to mode m and phase offset phi
                phases[mod_mask] = modulated[mod_mask] + phi
                self.slm.set_phases(phases)
                self.feedback.trigger(out=measurements[m, p, ...])

            if progress_bar is not None:
                progress_bar.update()

        self.feedback.wait()
        return self.analyzer(measurements, axis=1)

    @staticmethod
    def compute_t_set(wfs_result: WFSResult, mode_set: nd) -> nd:
        """
        Compute the transmission matrix in SLM space from transmission matrix in input mode space.

        Note 1: This function computes the transmission matrix for one mode set, and thus returns one part of the full
        transmission matrix. The elements that are not part of the mode set will be 0. The full transmission matrix can
        be obtained by simply adding the parts, i.e. t_full = t_set0 + t_set1.

        Note 2: As this is a blind focusing WFS algorithm, there may be only one target or 'output mode'.

        Args:
            wfs_result (WFSResult): The result of the WFS algorithm. This contains the transmission matrix in the space
                of input modes.
            mode_set: 3D array with set of modes.
        """
        t = wfs_result.t.squeeze().reshape((1, 1, mode_set.shape[2]))
        norm_factor = np.prod(mode_set.shape[0:2])
        t_set = (t * mode_set).sum(axis=2) / norm_factor
        return t_set
