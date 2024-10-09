from typing import Optional, List

import numpy as np
from numpy import ndarray as nd

from .utilities import analyze_phase_stepping, WFSResult, DummyProgressBar
from ..core import Detector, PhaseSLM


class DualReference:
    """A generic iterative dual reference WFS algorithm, which can use a custom set of basis functions.

    This algorithm is adapted from [Tao2017]_, with the addition of the ability to use custom basis functions and
    specify the number of iterations.

    In this algorithm, the SLM pixels are divided into two groups:
    A and B, as indicated by the boolean group_mask argument.
    The algorithm first keeps the pixels in group B fixed, and displays a sequence on patterns on the pixels of group A.
    It uses these measurements to construct an optimized wavefront that is displayed on the pixels of group A.
    This process is then repeated for the pixels of group B, now using the *optimized* wavefront on group A as
    reference. Optionally, the process can be repeated for a number of iterations, which each iteration using the current correction
    pattern as a reference. This makes this algorithm suitable for non-linear feedback, such as multi-photon
    excitation fluorescence [Osnabrugge2019]_.

    This algorithm assumes a phase-only SLM. Hence, the input modes are defined by passing the corresponding phase
    patterns (in radians) as input argument.

    References
    ----------
    .. [Tao2017] X. Tao, T. Lam, B. Zhu, et al., “Three-dimensional focusing through scattering media using conjugate
            adaptive optics with remote focusing (CAORF),” Opt. Express 25, 10368–10383 (2017).

    .. [Osnabrugge2019] Gerwin Osnabrugge, Lyubov V. Amitonova, and Ivo M. Vellekoop. "Blind focusing through strongly
            scattering media using wavefront shaping with nonlinear feedback", Optics Express, 27(8):11673–11688, 2019.
            https://opg.optica.org/oe/ abstract.cfm?uri=oe-27-8-1167

    """

    def __init__(
        self,
        *,
        feedback: Detector,
        slm: PhaseSLM,
        phase_patterns: Optional[tuple[nd, nd]],
        group_mask: nd,
        amplitude: nd = 1.0,
        phase_steps: int = 4,
        iterations: int = 2,
        optimized_reference: Optional[bool] = None
    ):
        """
        Args:
            feedback: The feedback source, usually a detector that provides measurement data.
            slm: Spatial light modulator object.
            phase_patterns:
                A tuple of two 3D arrays, containing the phase patterns for group A and group B, respectively.
                The 3D arrays have shape ``(pattern_count, height, width)``.
                The number of phase patterns in A and B may be different.
            amplitude:
                2D amplitude distribution on the SLM, should have shape ``(height, width)``.
            group_mask: A 2D bool array of that defines the pixels used by group A with False and elements used by
                group B with True, should have shape ``(height, width)``.
            phase_steps: The number of phase steps for each mode (default is 4). Depending on the type of
                non-linear feedback and the SNR, more might be required.
            iterations: Number of times to optimize a mode set, e.g. when iterations = 5, the measurements are
                A, B, A, B, A.
            optimized_reference:
                When `True`, during each iteration the other half of the SLM displays the optimized pattern so far (as in [1]).
                When `False`, the algorithm optimizes A with a flat wavefront on B,
                and then optimizes B with a flat wavefront on A.
                This mode also allows for multi-target optimization,
                where the algorithm optimizes multiple targets in parallel.
                The two halves are then combined (stitched) to form the full transmission matrix.
                In this mode, it is essential that both A and B include a flat wavefront as mode 0.
                The measurement for mode A0 and for B0 both give contain relative phase between group A and B,
                so there is a slight redundancy.
                These two measurements are combined to find the final phase for stitching.
                When set to `None` (default), the algorithm uses True if there is a single target,
                and False if there are multiple targets.

        [1]: X. Tao, T. Lam, B. Zhu, et al., “Three-dimensional focusing through scattering media using conjugate adaptive
        optics with remote focusing (CAORF),” Opt. Express 25, 10368–10383 (2017).
        """
        if optimized_reference is None:  # 'auto' mode
            optimized_reference = np.prod(feedback.data_shape) == 1
        elif optimized_reference and np.prod(feedback.data_shape) != 1:
            raise ValueError("In optimized_reference mode, only scalar (single target) feedback signals can be used.")
        if iterations < 2:
            raise ValueError("The number of iterations must be at least 2.")
        if not optimized_reference and iterations != 2:
            raise ValueError("When not using an optimized reference, the number of iterations must be 2.")

        self.slm = slm
        self.feedback = feedback
        self.phase_steps = phase_steps
        self.optimized_reference = optimized_reference
        self.iterations = iterations
        self._phase_patterns = None
        self._gram = None
        self._shape = group_mask.shape
        mask = group_mask.astype(bool)
        self.masks = (
            ~mask,
            mask,
        )  # self.masks[0] is True for group A, self.masks[1] is True for group B
        self._amplitude = 1.0
        self.amplitude = amplitude
        self.phase_patterns = phase_patterns

    @property
    def amplitude(self) -> Optional[nd]:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray):
        if not np.isscalar(value) and value.shape != self._shape:
            raise ValueError("The amplitude and group mask must all have the same shape.")

        self._amplitude = value

    @property
    def phase_patterns(self) -> tuple[nd, nd]:
        return self._phase_patterns

    @phase_patterns.setter
    def phase_patterns(self, value: np.ndarray):
        """Sets the phase patterns for group A and group B. This also updates the conjugate modes."""
        self._zero_indices = [0, 0]
        self._phase_patterns = [None, None]
        self._cobasis = [None, None]
        self._gram = [None, None]

        for side in range(2):
            patterns = value[side]
            mask = self.masks[side]
            if patterns[0].shape != self._shape:
                raise ValueError("The phase patterns and group mask must all have the same shape.")
            if not self.optimized_reference:
                # find the modes in A and B that correspond to flat wavefronts with phase 0
                try:
                    self._zero_indices[side] = next(i for i, p in enumerate(patterns) if np.allclose(p, 0))
                except StopIteration:
                    raise "For multi-target optimization, the both sets must contain a flat wavefront with phase 0."

            self._phase_patterns[side] = patterns.astype(np.float32)

            # Computes the cobasis
            # As a basis matrix B is full rank, the cobasis is equivalent to the Moore-Penrose pseudo-inverse B⁺.
            # B⁺ = (B^* B)⁻¹ B^*
            # Where B is the basis matrix, ^* denotes the conjugate transpose, and ^(-1)
            # denotes the matrix inverse. Note: we store the cobasis in transposed form,
            # with shape (mode, y, x) to allow for easy
            # multiplication with the transmission matrix t_ba.
            basis = self.amplitude * mask * np.exp(1j * patterns)
            basis /= np.linalg.norm(basis, axis=(1, 2), keepdims=True)
            gram = np.tensordot(basis, basis.conj(), axes=((1, 2), (1, 2)))  # inner product (contracts x and y axis)
            self._cobasis[side] = np.tensordot(np.linalg.inv(gram), basis.conj(), 1)
            self._gram[side] = gram

    @property
    def cobasis(self) -> tuple[nd, nd]:
        """
        The cobasis corresponding to the given basis.

        Note: The cobasis is stored in transposed form, with shape = (mode_count, height, width)
        """
        return tuple(self._cobasis)

    @property
    def gram(self) -> tuple[nd, nd]:
        """
        The Gram matrix corresponding to the given basis (i.e. phase pattern and amplitude profile).
        """
        return tuple(self._gram)

    def execute(self, *, capture_intermediate_results: bool = False, progress_bar=DummyProgressBar()) -> WFSResult:
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
        ref_phases = np.zeros(self._shape)

        # Initialize storage lists
        results_all: List[Optional[WFSResult]] = [None] * self.iterations  # List to store all results
        intermediate_results = np.zeros(self.iterations)  # List to store feedback from full patterns

        # Prepare progress bar
        progress_bar.total = (
            np.ceil(self.iterations / 2) * self.phase_patterns[0].shape[2]
            + np.floor(self.iterations / 2) * self.phase_patterns[1].shape[2]
        )

        # Switch the phase sets back and forth multiple times
        for it in range(self.iterations):
            side = it % 2  # pick set A or B for phase stepping
            side_mask = self.masks[side]
            # Perform WFS experiment on one side, keeping the other side sized at the ref_phases
            results_all[it] = self._single_side_experiment(
                mod_phases=self.phase_patterns[side],
                ref_phases=ref_phases,
                mod_mask=side_mask,
                progress_bar=progress_bar,
            )

            # Compute transmission matrix for the current side and update
            # estimated transmission matrix

            if self.optimized_reference:
                # use the best estimate so far to construct an optimized reference
                # TODO: see if the squeeze can be removed
                t_this_side = self._compute_t_set(results_all[it].t, side).squeeze()
                ref_phases[self.masks[side]] = -np.angle(t_this_side[self.masks[side]])

            # Try full pattern
            if capture_intermediate_results:
                self.slm.set_phases(ref_phases)
                intermediate_results[it] = self.feedback.read()

        if self.iterations % 2 == 0:
            t_side_0 = results_all[-2].t
            t_side_1 = results_all[-1].t
        else:
            t_side_0 = results_all[-1].t
            t_side_1 = results_all[-2].t

        if self.optimized_reference:
            factor = 1.0
        else:
            # when not using optimized reference, we need to stitch the
            # two halves of the wavefront together. For that, we need the
            # relative phase between the two sides, which we extract from
            # the measurements of the flat wavefronts.
            relative = t_side_0[..., self._zero_indices[0]] + np.conjugate(t_side_1[..., self._zero_indices[1]])
            factor = np.expand_dims(relative / np.abs(relative), -1)

        t_full = self._compute_t_set(t_side_0, 0) + self._compute_t_set(factor * t_side_1, 1)

        # Compute average fidelity factors
        # subtract 1 from n, because both sets (usually) contain a flat wavefront,
        # so there is one redundant measurement
        result = WFSResult.combine(results_all[-2:])
        result.n = result.n - 1
        result.t = t_full

        # TODO: document the results_all attribute
        result.results_all = results_all
        result.intermediate_results = intermediate_results
        return result

    def _single_side_experiment(self, mod_phases: nd, ref_phases: nd, mod_mask: nd, progress_bar) -> WFSResult:
        """
        Conducts experiments on one part of the SLM.

        Args:
            mod_phases: 3D array containing the phase patterns of each mode.
                ``shape = mode_count × height × width``
            ref_phases: 2D array containing the reference phase pattern.
            mod_mask: 2D array containing a boolean mask, where True indicates the modulated part of the SLM.
            progress_bar: Progress bar object. Following the convention for tqdm progress bars,
                this object should have a `total` attribute and an `update()` function.

        Returns:
            WFSResult: An object containing the computed SLM transmission matrix and related data.
        """
        measurements = np.zeros((len(mod_phases), self.phase_steps, *self.feedback.data_shape))

        for m, modulated in enumerate(mod_phases):
            phases = ref_phases.copy()
            for p in range(self.phase_steps):
                phi = p * 2 * np.pi / self.phase_steps
                # set the modulated pixel values to the values corresponding to mode m and phase offset phi
                phases[mod_mask] = modulated[mod_mask] + phi
                self.slm.set_phases(phases)
                self.feedback.trigger(out=measurements[m, p, ...])
                progress_bar.update()

        self.feedback.wait()
        return analyze_phase_stepping(measurements, axis=1)

    def _compute_t_set(self, t, side) -> nd:
        """
        Compute the transmission matrix in SLM space from transmission matrix in input mode space.

        Equivalent to ``np.tensordot(t, cobasis[side], 1)``

        Args:
            t: transmission matrix in mode-index space. The first axis corresponds to the input modes.
        Returns:
            nd: The transmission matrix in SLM space. The last two axes correspond to SLM coordinates
        """
        return np.tensordot(t, self.cobasis[side], 1)
