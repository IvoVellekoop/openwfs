from typing import Optional

import numpy as np
from numpy import ndarray as nd

from .utilities import analyze_phase_stepping, WFSResult
from ..core import Detector, PhaseSLM


class DualReference:
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
        *,
        feedback: Detector,
        slm: PhaseSLM,
        phase_patterns: Optional[tuple[nd, nd]],
        amplitude: Optional[tuple[nd, nd] | str],
        group_mask: nd,
        phase_steps: int = 4,
        iterations: int = 2,
        analyzer: Optional[callable] = analyze_phase_stepping,
        optimized_reference: Optional[bool] = None
    ):
        """
        Args:
            feedback: The feedback source, usually a detector that provides measurement data.
            slm: Spatial light modulator object.
            phase_patterns: A tuple of two 3D arrays, containing the phase patterns for group A and group B, respectively.
                The first two dimensions are the spatial dimensions, and should match the size of group_mask.
                The 3rd dimension in the array is index of the phase pattern. The number of phase patterns in A and B may be different.
                When None, the phase_patterns attribute must be set before executing the algorithm.
            amplitude: Tuple of 2D arrays, one array for each group. The arrays have shape equal to the shape of
                group_mask. When None, the amplitude attribute must be set before executing the algorithm. When
                'uniform', a 2D array of normalized uniform values is used, such that ⟨A,A⟩=1, where ⟨.,.⟩ denotes the
                inner product and A is the amplitude profile per group. This corresponds to a uniform illumination of
                the SLM. Note: if the groups have different sizes, their normalization factors will be different.
            group_mask: A 2D bool array of that defines the pixels used by group A with False and elements used by
                group B with True.
            phase_steps: The number of phase steps for each mode (default is 4). Depending on the type of
                non-linear feedback and the SNR, more might be required.
            iterations: Number of times to optimize a mode set, e.g. when iterations = 5, the measurements are
                A, B, A, B, A.
            optimized_reference: When `True`, during each iteration the other half of the SLM displays the optimized pattern so far (as in [1]).
                When `False`, the algorithm optimizes A with a flat wavefront on B, and then optimizes B with a flat wavefront on A.
                This mode also allows for multi-target optimization, where the algorithm optimizes multiple targets in parallel.
                The two halves are then combined (stitched) to form the full transmission matrix.
                In this mode, it is essential that both A and B include a flat wavefront as mode 0. The measurement for
                mode A0 and for B0 both give contain relative phase between group A and B, so there is a slight redundancy.
                These two measurements are combined to find the final phase for stitching.
                When set to `None` (default), the algorithm uses True if there is a single target, and False if there are multiple targets.

            analyzer: The function used to analyze the phase stepping data.
                Must return a WFSResult object. Defaults to `analyze_phase_stepping`

        [1]: X. Tao, T. Lam, B. Zhu, et al., “Three-dimensional focusing through scattering media using conjugate adaptive
        optics with remote focusing (CAORF),” Opt. Express 25, 10368–10383 (2017).
        """
        if optimized_reference is None:  # 'auto' mode
            optimized_reference = np.prod(feedback.data_shape) == 1
        elif optimized_reference and np.prod(feedback.data_shape) != 1:
            raise ValueError(
                "When using an optimized reference, the feedback detector should return a single scalar value."
            )

        if iterations < 2:
            raise ValueError("The number of iterations must be at least 2.")
        if not optimized_reference and iterations != 2:
            raise ValueError(
                "When not using an optimized reference, the number of iterations must be 2."
            )

        self.slm = slm
        self.feedback = feedback
        self.phase_steps = phase_steps
        self.optimized_reference = optimized_reference
        self.iterations = iterations
        self._analyzer = analyzer
        self._phase_patterns = None
        self._amplitude = None
        self._gram = None
        self._shape = group_mask.shape
        mask = group_mask.astype(bool)
        self.masks = (
            ~mask,
            mask,
        )  # self.masks[0] is True for group A, self.masks[1] is True for group B
        self.amplitude = amplitude      # Note: when 'uniform' is passed, the shape of self.masks[0] is used.
        self.phase_patterns = phase_patterns

    @property
    def amplitude(self) -> Optional[nd]:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if value is None:
            self._amplitude = None
            return

        if value == 'uniform':
            self._amplitude = tuple(
                (np.ones(shape=self._shape) / np.sqrt(self.masks[side].sum())).astype(np.float32) for side in range(2))
            return

        if value.shape != self._shape:
            raise ValueError(
                "The amplitude and group mask must all have the same shape."
            )

        self._amplitude = value.astype(np.float32)


    @property
    def phase_patterns(self) -> tuple[nd, nd]:
        return self._phase_patterns

    @phase_patterns.setter
    def phase_patterns(self, value):
        """Sets the phase patterns for group A and group B. This also updates the conjugate modes."""
        if value is None:
            self._phase_patterns = None
            return

        if not self.optimized_reference:
            # find the modes in A and B that correspond to flat wavefronts with phase 0
            try:
                a0_index = next(
                    i
                    for i in range(value[0].shape[2])
                    if np.allclose(value[0][:, :, i], 0)
                )
                b0_index = next(
                    i
                    for i in range(value[1].shape[2])
                    if np.allclose(value[1][:, :, i], 0)
                )
                self.zero_indices = (a0_index, b0_index)
            except StopIteration:
                raise (
                    "For multi-target optimization, the both sets must contain a flat wavefront with phase 0."
                )

        if (value[0].shape[0:2] != self._shape) or (value[1].shape[0:2] != self._shape):
            raise ValueError(
                "The phase patterns and group mask must all have the same shape."
            )

        self._phase_patterns = (
            value[0].astype(np.float32),
            value[1].astype(np.float32),
        )

        self._compute_cobasis()

    @property
    def cobasis(self) -> tuple[nd, nd]:
        """
        The cobasis corresponding to the given basis.
        """
        return self._cobasis

    @property
    def gram(self) -> np.matrix:
        """
        The Gram matrix corresponding to the given basis (i.e. phase pattern and amplitude profile).
        """
        return self._gram

    def _compute_cobasis(self):
        """
        Computes the cobasis from the phase patterns.

        As a basis matrix is full rank, this is equivalent to the Moore-Penrose pseudo-inverse.
        B⁺ = (B^* B)^(-1) B^*
        Where B is the basis matrix (a column corresponds to a basis vector), ^* denotes the conjugate transpose, ^(-1)
        denotes the matrix inverse, and ⁺ denotes the Moore-Penrose pseudo-inverse.
        """
        if self.phase_patterns is None:
            raise('The phase_patterns must be set before computing the cobasis.')

        cobasis = [None, None]
        for side in range(2):
            p = np.prod(self._shape)  # Number of SLM pixels
            m = self.phase_patterns[side].shape[2]  # Number of modes
            phase_factor = np.exp(1j * self.phase_patterns[side])
            amplitude_factor = np.expand_dims(self.amplitude[side] * self.masks[side], axis=2)
            B = np.asmatrix((phase_factor * amplitude_factor).reshape((p, m)))  # Basis matrix
            self._gram = B.H @ B
            B_pinv = np.linalg.inv(self.gram) @ B.H  # Moore-Penrose pseudo-inverse
            cobasis[side] = np.asarray(B_pinv.T).reshape(self.phase_patterns[side].shape)

        self._cobasis = cobasis

    def execute(
        self, *, capture_intermediate_results: bool = False, progress_bar=None
    ) -> WFSResult:
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
        results_all = [None] * self.iterations  # List to store all results
        intermediate_results = np.zeros(
            self.iterations
        )  # List to store feedback from full patterns

        # Prepare progress bar
        if progress_bar:
            num_measurements = (
                np.ceil(self.iterations / 2) * self.phase_patterns[0].shape[2]
                + np.floor(self.iterations / 2) * self.phase_patterns[1].shape[2]
            )
            progress_bar.total = num_measurements

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
                t_this_side = self.compute_t_set(results_all[it].t, self.cobasis[side]).squeeze()
                ref_phases[self.masks[side]] = -np.angle(t_this_side[self.masks[side]])

            # Try full pattern
            if capture_intermediate_results:
                self.slm.set_phases(ref_phases)
                intermediate_results[it] = self.feedback.read()

        if self.optimized_reference:
            factor = 1.0
        else:
            # when not using optimized reference, we need to stitch the
            # two halves of the wavefront together. For that, we need the
            # relative phase between the two sides, which we extract from
            # the measurements of the flat wavefronts.
            relative = results_all[0].t[self.zero_indices[0], ...] + np.conjugate(
                results_all[1].t[self.zero_indices[1], ...]
            )
            factor = (relative / np.abs(relative)).reshape(
                (1, *self.feedback.data_shape)
            )

        t_full = self.compute_t_set(results_all[0].t, self.cobasis[0]) + self.compute_t_set(
            factor * results_all[1].t, self.cobasis[1]
        )

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

    def _single_side_experiment(
        self, mod_phases: nd, ref_phases: nd, mod_mask: nd, progress_bar=None
    ) -> WFSResult:
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
        measurements = np.zeros(
            (num_modes, self.phase_steps, *self.feedback.data_shape)
        )

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
        return self._analyzer(measurements, axis=1)

    @staticmethod
    def compute_t_set(t, cobasis: nd) -> nd:
        """
        Compute the transmission matrix in SLM space from transmission matrix in input mode space.

        Note 1: This function computes the transmission matrix for one mode set, and thus returns one part of the full
        transmission matrix. The elements that are not part of the mode set will be 0. The full transmission matrix can
        be obtained by simply adding the parts, i.e. t_full = t_set0 + t_set1.

        Note 2: As this is a blind focusing WFS algorithm, there may be only one target or 'output mode'.

        Args:
            t: transmission matrix in mode-index space. The first axis corresponds to the input modes.
            cobasis: 3D array with set of modes (conjugated)
        Returns:
            nd: The transmission matrix in SLM space. The last two axes correspond to SLM coordinates
        """
        norm_factor = np.prod(cobasis.shape[0:2])
        return np.tensordot(cobasis, t, 1) / norm_factor
