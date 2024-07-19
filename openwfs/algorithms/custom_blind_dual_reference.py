from typing import Optional

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
    focusing [2].

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering", Optics Express 30, 37436-37445 (2022)
    [2]: Gerwin Osnabrugge, Lyubov V. Amitonova, and Ivo M. Vellekoop. "Blind focusing through strongly scattering media
    using wavefront shaping with nonlinear feedback", Optics Express, 27(8):11673–11688, 2019.
    https://opg.optica.org/oe/ abstract.cfm?uri=oe-27-8-1167
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, slm_shape: tuple[int, int], modes: tuple[nd, nd],
                 phase_steps: int = 4, iterations: int = 4, analyzer: Optional[callable] = analyze_phase_stepping,
                 do_try_full_patterns=False, do_progress_bar=True, progress_bar_kwargs={}):
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
            modes (tuple): A tuple of two 3D arrays. We will refer to these as set A and B. The 3D arrays contain the
                set of modes (complex fields) to measure side A, B respectively. From these 3D arrays, axis 0 and 1 are
                used as spatial axes. Axis 2 is used as mode index. E.g. modes[1][:, :, 4] is the 4th mode of set B.
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

        assert (modes[0].shape[0] == modes[1].shape[0]) and (modes[0].shape[1] == modes[1].shape[1])

        self.modes = modes
        self.phases = (np.angle(modes[0]), np.angle(modes[1]))      # Pre-compute the phases of each mode

    def execute(self) -> WFSResult:
        """
        Executes the FourierDualRef algorithm, computing the SLM transmission matrix.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """

        # Initial transmission matrix for reference is constant phase
        t_full = np.zeros(shape=self.modes[0].shape[0:2])
        t_set = np.zeros(shape=self.modes[0].shape[0:2])

        # Initialize storage lists
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
            t_prev = t_set
            ref_phases = -np.angle(t_prev)  # Shaped reference pattern from transmission matrix
            set = it % 2  # Pick set A or B

            # Measure and compute
            result = self._single_side_experiment(self.phases[set], ref_phases=ref_phases, progress_bar=progress_bar)
            t_set = self.compute_t_set(result, self.modes[set])  # Compute transmission matrix from measurements

            # Store results
            t_set_all[it] = t_set  # Store transmission matrix
            results_all[it] = result  # Store result
            results_latest[set] = result  # Store latest result for this set
            t_full = t_prev + t_set

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

    def _single_side_experiment(self, step_phases: nd, ref_phases, progress_bar: Optional[tqdm] = None) -> WFSResult:

        ################################################# Fix everything below
        """
        Conducts experiments on one side of the SLM, generating measurements for each spatial frequency and phase step.

        Args:
            k_set (np.ndarray): An array of spatial frequencies to use in the experiment.
            side (int): Indicates which side of the SLM to use (0 for the left hand side, 1 for right hand side).

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        measurements = np.zeros((k_set.shape[1], self.phase_steps, *self.feedback.data_shape))

        for i in range(k_set.shape[1]):
            for p in range(self.phase_steps):
                phase_offset = p * 2 * np.pi / self.phase_steps
                phase_pattern = self._get_phase_pattern_full(k_set[:, i], phase_offset, side, reference_pattern)

                self.slm.set_phases(phase_pattern)
                self.feedback.trigger(out=measurements[i, p, ...])

            if progress_bar is not None:
                progress_bar.update()

        self.feedback.wait()
        return self.analyzer(measurements, axis=1)

        def _get_phase_pattern_half(self, k: nd, phase_offset: float, side: int):
            """
            Generates a phase pattern for one half of the SLM, based on the given spatial frequency and phase offset.

            Args:
                k (np.ndarray): A 2-element array representing the spatial frequency.
                phase_offset (float): The phase offset to apply to the pattern.

            Returns:
                np.ndarray: The generated phase pattern for one half.
            """
            # tilt generates a pattern from -2.0 to 2.0 (The convention for Zernike modes normalized to an RMS of 1).
            # The natural step to take is the Abbe diffraction limit, which corresponds to a gradient from -π to π.
            return tilt([self.slm_shape[0], self.slm_shape[1] // 2], k * (0.5 * np.pi), extent=(2.0, 1.0),
                        phase_offset=phase_offset)

        def _get_phase_pattern_full(self, k: nd, phase_offset: float, side: int, reference_pattern: nd) -> nd:
            """
            Generates a phase pattern for the SLM based on the given spatial frequency, phase offset, and side.

            Args:
                k (np.ndarray): A 2-element array representing the spatial frequency of the whole pupil.
                phase_offset (float): The phase offset to apply to the pattern.
                side (int): Indicates the side of the SLM for the pattern (0 for left, 1 for right).

            Returns:
                np.ndarray: The generated phase pattern.
            """
            tilted_front = self._get_phase_pattern_half(k, phase_offset, side)
            # Concatenate based on the side
            if side == 0:  # Place the pattern on the left
                result = np.concatenate((tilted_front, reference_pattern), axis=1)
            else:  # Place the pattern on the right
                result = np.concatenate((reference_pattern, tilted_front), axis=1)

            return result

        def compute_t_half(self, wfs_result: WFSResult, k: nd, side: int):
            """
            Compute the transmission matrix for a single half of a FourierDualReference algorithm.

            Args:
                wfs_result (WFSResult): The result of the WFS algorithm.
                k (np.ndarray): The spatial frequencies used for the measurements.
                side (int): The side of the SLM to compute the transmission matrix for.
            """
            t_sum = np.zeros((self.slm_shape[0], self.slm_shape[1] // 2, *self.feedback.data_shape), dtype='complex128')

            # TODO: Change enumerate in all Fourier WFS versions, so that it supports multiple targets
            for n, t_current in enumerate(wfs_result.t):  # Sum all Fourier modes
                phi = self._get_phase_pattern_half(k[:, n], 0, side=side)
                t_sum += np.tensordot(np.exp(-1j * phi), t_current, 0)

            t = t_sum / (0.5 * self.slm_shape[0] * self.slm_shape[1])  # Normalize the Fourier transform

            return t

        @property
        def k_radius(self) -> float:
            """The maximum radius of the k-space circle."""
            return self._k_radius

        @k_radius.setter
        def k_radius(self, value):
            """Don't set this value directly."""
            pass

    def _get_phase_pattern_half(self, k: nd, phase_offset: float, side: int):
        """
        Generates a phase pattern for one half of the SLM, based on the given spatial frequency and phase offset.

        Args:
            k (np.ndarray): A 2-element array representing the spatial frequency.
            phase_offset (float): The phase offset to apply to the pattern.

        Returns:
            np.ndarray: The generated phase pattern for one half.
        """
        # tilt generates a pattern from -2.0 to 2.0 (The convention for Zernike modes normalized to an RMS of 1).
        # The natural step to take is the Abbe diffraction limit, which corresponds to a gradient from -π to π.
        field = compute_mode(mode_shape=self.slm_shape, k=k, r_factor=None, ax=self.warp_coeffs[0],
                             ay=self.warp_coeffs[1], ignore_warp=False, ignore_amplitude=True)
        phase_factor = np.exp(1j * phase_offset)
        phase_pattern_half = (field * phase_factor).angle().squeeze().numpy()
        if side == 0:
            return phase_pattern_half
        else:
            return np.flip(phase_pattern_half, axis=1)

    def _get_phase_pattern_full(self, k: nd, phase_offset: float, side: int, reference_pattern: nd) -> nd:
        """
        Generates a phase pattern for the SLM based on the given spatial frequency, phase offset, and side.

        Args:
            k (np.ndarray): A 2-element array representing the spatial frequency of the whole pupil.
            phase_offset (float): The phase offset to apply to the pattern.
            side (int): Indicates the side of the SLM for the pattern (0 for A, 1 for B).

        Returns:
            np.ndarray: The generated phase pattern.
        """
        tilted_front = self._get_phase_pattern_half(k, phase_offset, side)

        # Concatenate based on the side
        if side == 0:  # Place the pattern on the A
            result = np.concatenate((tilted_front, reference_pattern), axis=1)
        else:  # Place the pattern on the B
            result = np.concatenate((reference_pattern, tilted_front), axis=1)

        return result
