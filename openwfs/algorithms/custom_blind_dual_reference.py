from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray as nd
from tqdm import tqdm

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


class CustomBlindDualReference:
    """
    A generic blind focusing dual reference WFS algorithm.

    This generic WFS algorithm uses two sets of modes (set A and set B), similar to the Fourier Dual Reference method as
    described in [1]. First, set A is used for phase stepping and set with a flat reference. We construct a correction
    pattern from the A measurements. Then, we phase step B and use the A correction pattern as reference. This is
    repeated to converge to a correction even with a non-local, non-linear feedback signal. This is known as blind
    focusing [2]. This algorithm assumes a phase-only SLM. The input modes are defined by passing the corresponding
    phase patterns to the algorithm.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering", Optics Express 30, 37436-37445 (2022)
    [2]: Gerwin Osnabrugge, Lyubov V. Amitonova, and Ivo M. Vellekoop. "Blind focusing through strongly scattering media
    using wavefront shaping with nonlinear feedback", Optics Express, 27(8):11673–11688, 2019.
    https://opg.optica.org/oe/ abstract.cfm?uri=oe-27-8-1167
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape: tuple[int, int], phases: tuple[nd, nd], set1_mask:
                 nd, phase_steps: int = 4, iterations: int = 4, analyzer: Optional[callable] = analyze_phase_stepping,
                 do_try_full_patterns=False, do_progress_bar=True, progress_bar_kwargs={}):
        #### TODO: Actually, we don't want to cut off the pattern at the pupil edge, just in case of misalignment.
        #### TODO: Moreover, the amplitude profile doesn't really add anything. You can compute the same pattern without
        #### TODO: it, and multiply with it later if you want to.
        """
        Args:
            feedback (Detector): The feedback source, usually a detector that provides measurement data.
            slm (PhaseSLM): slm object.
                The slm may have the `extent` property set to indicate the extent of the back pupil of the microscope
                objective in slm coordinates. By default, a value of 2.0, 2.0 is used (indicating that the pupil
                corresponds to a circle of radius 1.0 on the SLM). However, to prevent artefacts at the edges of the
                SLM,it may be overfilled, such that the `phases` image is mapped to an extent of e. g. (2.2, 2.2),
                i. e. 10% larger than the back pupil.
            slm_shape (tuple[int, int]): The shape of the SLM patterns and transmission matrices.
            phases (tuple): A tuple of two 3D arrays. We will refer to these as set A and B (phases[0] and
                phases[1] respectively). The 3D arrays contain the set of phases to measure set A and B. With both of
                these 3D arrays, axis 0 and 1 are used as spatial axes. Axis 2 is used as phase pattern index.
                E.g. phases[1][:, :, 4] is the 4th phase pattern of set B.
            set1_mask: A 2D array of that defines the elements used by set A (= modes[0]) with 0s and elements used by
                set B (= modes[1]) with 1s.
            phase_steps (int): The number of phase steps for each mode (default is 4). Depending on the type of
                non-linear feedback. More might be required.
            iterations (int): Number of times to measure a mode set, e.g. when iterations = 5, the measurements are
                A, B, A, B, A.
            analyzer (callable): The function used to analyze the phase stepping data. Must return a WFSResult object.
            do_try_full_patterns (bool): Whether to measure feedback from the full patterns each iteration. This can
                be useful to determine how many iterations are needed to converge to an optimal pattern.
        """
        self.slm = slm
        self.feedback = feedback
        self.phase_steps = phase_steps
        self.iterations = iterations
        self.slm_shape = slm_shape
        self.analyzer = analyzer
        self.do_try_full_patterns = do_try_full_patterns
        self.do_progress_bar = do_progress_bar
        self.progress_bar_kwargs = progress_bar_kwargs

        assert (phases[0].shape[0] == phases[1].shape[0]) and (phases[0].shape[1] == phases[1].shape[1])
        self.phases = (phases[0].astype(np.float32), phases[1].astype(np.float32))

        # Pre-compute set0 mask
        mask1 = set1_mask.astype(dtype=np.float32)
        mask0 = 1.0 - mask1
        self.set_masks = (mask0, mask1)

        # Pre-compute the conjugate modes for reconstruction
        modes0 = np.exp(-1j * self.phases[0]) * np.expand_dims(mask0, axis=2)
        modes1 = np.exp(-1j * self.phases[1]) * np.expand_dims(mask1, axis=2)
        self.modes = (modes0, modes1)

    def execute(self) -> WFSResult:
        """
        Executes the FourierDualRef algorithm, computing the SLM transmission matrix.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data. The amplitude profile
                of each mode is assumed to be 1. If a different amplitude profile is desired, this can be obtained by
                multiplying that amplitude profile with this transmission matrix.
        """

        # Initial transmission matrix for reference is constant phase
        t_full = np.zeros(shape=self.modes[0].shape[0:2])

        # Initialize storage lists
        t_set = t_full
        t_set_all = [None] * self.iterations
        results_all = [None] * self.iterations  # List to store all results
        results_latest = [None, None]  # The two latest results. Used for computing fidelity factors.
        full_pattern_feedback = np.zeros(self.iterations)  # List to store feedback from full patterns

        # Prepare progress bar
        if self.do_progress_bar:
            num_measurements = np.ceil(self.iterations / 2) * self.modes[0].shape[2] \
                            + np.floor(self.iterations / 2) * self.modes[1].shape[2]
            progress_bar = tqdm(total=num_measurements, **self.progress_bar_kwargs)
        else:
            progress_bar = None

        for it in range(self.iterations):
            set = it % 2  # Pick set A or B for phase stepping
            step_mask = self.set_masks[set]
            t_prev = t_set
            ref_phases = -np.angle(t_prev)  # Shaped reference pattern from transmission matrix

            # Measure and compute
            result = self._single_side_experiment(step_phases=self.phases[set], ref_phases=ref_phases,
                                                  step_mask=step_mask, progress_bar=progress_bar)
            t_set = self.compute_t_set(result, self.modes[set])  # Compute transmission matrix from measurements

            # Store results
            t_full = t_prev + t_set
            t_set_all[it] = t_set  # Store transmission matrix
            results_all[it] = result  # Store result
            results_latest[set] = result  # Store latest result for this set

            # Try full pattern
            if self.do_try_full_patterns:
                self.slm.set_phases(-np.angle(t_full))
                full_pattern_feedback[it] = self.feedback.read()

        # Compute average fidelity factors
        fidelity_noise = weighted_average(results_latest[0].fidelity_noise,
                                          results_latest[1].fidelity_noise, results_latest[0].n,
                                          results_latest[1].n)
        fidelity_amplitude = weighted_average(results_latest[0].fidelity_amplitude,
                                              results_latest[1].fidelity_amplitude, results_latest[0].n,
                                              results_latest[1].n)
        fidelity_calibration = weighted_average(results_latest[0].fidelity_calibration,
                                                results_latest[1].fidelity_calibration, results_latest[0].n,
                                                results_latest[1].n)

        result = WFSResult(t=t_full,
                           t_f=None,
                           n=self.modes[0].shape[2]+self.modes[1].shape[2],
                           axis=2,
                           fidelity_noise=fidelity_noise,
                           fidelity_amplitude=fidelity_amplitude,
                           fidelity_calibration=fidelity_calibration)

        # TODO: This is a dirty way to add attributes. Find better way.
        result.t_set_all = t_set_all
        result.results_all = results_all
        result.full_pattern_feedback = full_pattern_feedback
        return result

    def _single_side_experiment(self, step_phases: nd, ref_phases: nd, step_mask: nd,
                                progress_bar: Optional[tqdm] = None) -> WFSResult:
        """
        Conducts experiments on one part of the SLM.

        Args:
            step_phases: 3D array containing the phase patterns of each mode. Axis 0 and 1 are used as spatial axis.
                Axis 2 is used for the mode index.
            ref_phases: 2D array containing the reference phase pattern.
            step_mask: 2D array containing a mask of 1s and 0s for the phase stepped modes.
            progress_bar: An optional tqdm progress bar.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.

        Note: In order to speed up calculations, I used np.float32 datatypes for the phases, with a mask, instead of
        adding the complex128 fields and taking the np.angle. I did a quick test for this (on AMD Ryzen 7 3700X) for a
        1000x1000 array, this brings down the phase pattern computation time from ~26ms (with complex128:
        np.angle(field_B + field_A * np.exp(step))) to ~2ms (with float32: phases_B + (phases_A + step) * mask).
        """
        num_of_modes = step_phases.shape[2]
        measurements = np.zeros((num_of_modes, self.phase_steps, *self.feedback.data_shape))
        ref_phases_masked = (1.0 - step_mask) * ref_phases

        for m in range(num_of_modes):
            for p in range(self.phase_steps):
                phase_step = p * 2 * np.pi / self.phase_steps
                phase_pattern = ref_phases_masked + step_mask * (step_phases[:, :, m] + phase_step)
                self.slm.set_phases(phase_pattern)
                self.feedback.trigger(out=measurements[m, p, ...])

                # ###############
                # plt.clf()
                # plt.subplot(1, 4, 1)
                # plt.imshow(ref_phases_masked, vmin=-np.pi, vmax=np.pi)
                #
                # plt.subplot(1,4,2)
                # plt.imshow(step_phases[:, :, m], vmin=-np.pi, vmax=np.pi)
                #
                # plt.subplot(1,4,3)
                # plt.imshow(step_mask)
                #
                # plt.subplot(1,4,4)
                # plt.imshow(self.slm.pixels.read(), vmin=0, vmax=255)
                # # plt.imshow(phase_pattern, vmin=-np.pi, vmax=np.pi)
                # plt.title(f'm={m}, p={p}')
                # plt.pause(0.02)
                # ###############

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
