from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


class WFSResult:
    """
    Data structure for holding wavefront shaping results and statistics.

    Attributes:
        t (ndarray): Measured transmission matrix. If multiple targets were used, the first dimension(s) of `t`
            denote the columns of the transmission matrix (`a` indices), and the last dimensions(s) denote the targets,
            i.e., the rows of the transmission matrix (`b` indices).
        axis (int): Number of dimensions used for denoting a single column of the transmission matrix
            (e.g., 2 dimensions representing the x and y coordinates of the SLM pixels).
        fidelity_noise (ndarray): The estimated loss in fidelity caused by the limited SNR (for each target).
        fidelity_amplitude (ndarray): Estimated reduction of the fidelity due to phase-only modulation (for each target)
            (≈ π/4 for fully developed speckle).
        fidelity_calibration (ndarray): Estimated deviation from a sinusoid response.
        n (int): Total number of segments used in the optimization. When missing, this value is set to the number of
            elements in the first `axis` dimensions of `t`.
        estimated_optimized_intensity (ndarray): When missing, estimated intensity in the target(s) after displaying the
            wavefront correction on a perfect phase-only SLM.
        intensity_offset (Optional[ndarray]): Offset in the signal strength, as a scalar, or as one value per target.
            This is the offset that is caused by a bias in the detector signal, stray light, etc. Default value: 0.0.
    """

    def __init__(
        self,
        t: np.ndarray,
        axis: int,
        fidelity_noise: ArrayLike = 1.0,
        fidelity_amplitude: ArrayLike = 1.0,
        fidelity_calibration: ArrayLike = 1.0,
        n: Optional[int] = None,
        intensity_offset: Optional[ArrayLike] = 0.0,
    ):
        """
        Args:
            t(ndarray): measured transmission matrix.
            axis(int):
                number of dimensions used for denoting a single columns of the transmission matrix
                (e.g. 2 dimensions representing the x and y coordinates of the SLM pixels)
            fidelity_noise(ArrayLike):
                the estimated loss in fidelity caused by the the limited snr (for each target).
            fidelity_amplitude(ArrayLike):
                estimated reduction of the fidelity due to phase-only modulation (for each target)
                (≈ π/4 for fully developed speckle)
            fidelity_calibration(ArrayLike):
                estimated deviation from a sinusoid responds (TODO: experimental, untested)
            n(Optional[int]): total number of segments used in the optimization.
                when missing, this value is set to the number of elements in the first `axis` dimensions of `t`.
            intensity_offset(Optional[ArrayLike]):
                offset in the signal strength, as a scalar, or as one value per target
                this is the offset that is caused by a bias in the detector signal, stray light, etc.
                default value: 0.0

        """
        self.t = t
        self.axis = axis
        self.fidelity_noise = np.atleast_1d(fidelity_noise)
        self.n = np.prod(t.shape[0:axis]) if n is None else n
        self.fidelity_amplitude = np.atleast_1d(fidelity_amplitude)
        self.fidelity_calibration = np.atleast_1d(fidelity_calibration)
        self.estimated_enhancement = np.atleast_1d(
            1.0 + (self.n - 1) * self.fidelity_amplitude * self.fidelity_noise * self.fidelity_calibration
        )
        self.intensity_offset = (
            intensity_offset * np.ones(self.fidelity_calibration.shape)
            if np.isscalar(intensity_offset)
            else intensity_offset
        )
        after = (
            np.sum(np.abs(t), tuple(range(self.axis))) ** 2 * self.fidelity_noise * self.fidelity_calibration
            + intensity_offset
        )
        self.estimated_optimized_intensity = np.atleast_1d(after)

    def __str__(self) -> str:
        noise_warning = "OK" if self.fidelity_noise > 0.5 else "WARNING low signal quality."
        amplitude_warning = "OK" if self.fidelity_amplitude > 0.5 else "WARNING uneven contribution of optical modes."
        calibration_fidelity_warning = (
            "OK" if self.fidelity_calibration > 0.5 else "WARNING non-linear phase response, check " "lookup table."
        )
        return f"""
        Wavefront shaping results:
            fidelity_noise: {self.fidelity_noise} {noise_warning}
            fidelity_amplitude: {self.fidelity_amplitude} {amplitude_warning}
            fidelity_calibration: {self.fidelity_calibration} {calibration_fidelity_warning}
            estimated_enhancement: {self.estimated_enhancement}
            estimated_optimized_intensity: {self.estimated_optimized_intensity}
            """

    def select_target(self, b) -> "WFSResult":
        """
        Returns the wavefront shaping results for a single target

        Args:
            b(int): target to select, as integer index.
                If the target array is multidimensional, it is flattened before selecting the `b`-th component.

        Returns: WFSResults data for the specified target
        """
        return WFSResult(
            t=self.t.reshape((*self.t.shape[0:2], -1))[:, :, b],
            axis=self.axis,
            intensity_offset=self.intensity_offset[:][b],
            fidelity_noise=self.fidelity_noise[:][b],
            fidelity_amplitude=self.fidelity_amplitude[:][b],
            fidelity_calibration=self.fidelity_calibration[:][b],
            n=self.n,
        )

    @staticmethod
    def combine(results: Sequence["WFSResult"]):
        """Merges the results for several sub-experiments.

        Currently, this just computes the average of the fidelities, weighted
        by the number of segments used in each sub-experiment.

        Note: the matrix t is also averaged, but this is not always meaningful.
        The caller can replace the `.t` attribute of the result with a more meaningful value.
        """
        n = sum(r.n for r in results)
        axis = results[0].axis
        if any(r.axis != axis for r in results):
            raise ValueError("All results must have the same axis")

        def weighted_average(attribute):
            data = getattr(results[0], attribute) * results[0].n / n
            for r in results[1:]:
                data += getattr(r, attribute) * r.n / n
            return data

        return WFSResult(
            t=weighted_average("t"),
            n=n,
            axis=axis,
            fidelity_noise=weighted_average("fidelity_noise"),
            fidelity_amplitude=weighted_average("fidelity_amplitude"),
            fidelity_calibration=weighted_average("fidelity_calibration"),
        )

    @property
    def snr(self):
        return 1.0 / (1.0 / self.fidelity_noise - 1.0)


def analyze_phase_stepping(measurements: np.ndarray, axis: int):
    """Analyzes the result of phase stepping measurements, returning matrix `t` and noise statistics

    This function assumes that all measurements were made using the same reference field `A`
    and that the phase of the modulated segment/mode is phase-stepped in equally spaced steps
    between 0 and 2π, whereas the phase of the reference field is kept constant.

    Args:
        measurements(ndarray): array of phase stepping measurements.
            The array holds measured intensities
            with the first one or more dimensions (a1, a2, ...) corresponding to the segments of the SLM,
            one dimension corresponding to the phase steps,
            and the last zero or more dimensions (b1, b2, ...) corresponding to the individual targets
            where the feedback was measured.
        axis(int): indicates which axis holds the phase steps.

    With `phase_steps` phase steps, the measurements are given by

    .. math::

        I_p = \\lvert A + B \\exp(i 2\\pi p / phase_{steps})\\rvert^2,

    This function computes the Fourier transform.

    .. math::

        \\frac{1}{phase_{steps}} \\sum I_p  \\exp(-i 2\\pi p / phase_{steps}) = A^* B

    Returns:
        WFSResult: The result of the analysis. The attribute `t` holds the complex transmission matrix.
            Note that the dimensions of t are reversed with respect to the input, so t has shape b1×b2×...×a1×a2×...
            Other attributes hold fidelity estimates (see WFSResult).
    """
    phase_steps = measurements.shape[axis]
    a_count = int(np.prod(measurements.shape[:axis]))
    a_axes = tuple(range(axis))

    # Fourier transform the phase stepping measurements
    t_f = np.fft.fft(measurements, axis=axis) / phase_steps
    t = np.take(t_f, 1, axis=axis)

    # compute the effect of amplitude variations.
    # for perfectly developed speckle, and homogeneous illumination, this factor will be pi/4
    fidelity_amplitude = np.mean(np.abs(t), a_axes) ** 2 / np.mean(np.abs(t) ** 2, a_axes)

    # estimate the calibration error
    # we first construct a matrix that can be used to fit
    # parameters a and b such that a·t(:) + b·t^*(:) ≈ t_f(:, k, :)
    ff = np.vstack((t.ravel(), np.conj(t.ravel()))).T
    ff_inv = np.linalg.pinv(ff)
    c = np.zeros(phase_steps, np.complex128)

    signal_energy = 0
    for k in range(1, phase_steps):
        cc = ff_inv @ np.take(t_f, k, axis=axis).ravel()
        signal_energy = signal_energy + np.sum(np.abs(ff @ cc) ** 2)
        c[k] = cc[0]
    fidelity_calibration = np.abs(c[1]) ** 2 / np.sum(np.abs(c[1:]) ** 2)

    # TODO: use the pinv fit to estimate the signal strength (requires special treatment of offset)

    # Estimate the error due to noise
    # The signal consists of the response with incorrect modulation,
    # (which occurs twice, ideally in the +1 and -1 components of the Fourier transform),
    # but this factor of two is already included in the 'signal_energy' calculation.
    # an offset, and the rest is noise.
    # average over all targets to get the most accurate result (assuming all targets are similar)
    axes = tuple([i for i in range(t_f.ndim) if i != axis])
    energies = np.sum(np.abs(t_f) ** 2, axis=axes)
    total_energy = np.sum(energies)
    offset_energy = energies[0]
    signal_energy = energies[1] + energies[-1]
    if phase_steps > 3:
        # estimate the noise energy as the energy that is not explained
        # by the signal or the offset.
        noise_energy = (total_energy - signal_energy - offset_energy) / (phase_steps - 3)
        fidelity_noise = np.abs(np.maximum(signal_energy - 2 * noise_energy, 0.0) / signal_energy)
    else:
        fidelity_noise = 1.0  # cannot estimate reliably

    # convert from t_ab to t_ba form
    a_destination = np.array(a_axes) + t.ndim - len(a_axes)
    t = np.moveaxis(t, a_axes, a_destination)

    return WFSResult(
        t,
        axis=axis,
        fidelity_amplitude=fidelity_amplitude,
        fidelity_noise=fidelity_noise,
        fidelity_calibration=fidelity_calibration,
        n=a_count,
    )


class WFSController:
    """
    EXPERIMENTAL - Controller for Wavefront Shaping (WFS) operations using a specified algorithm in the Micro-Manager environment.

    Usage:

    .. code-block:: python

        # not wrapped:
        alg = FourierDualReference(feedback, slm)

        # wrapped
        alg = WFSController(FourierDualReference, feedback, slm)

    Under the hood, a dynamic class is created that inherits both ``WFSController`` and ``FourierDualReference)``.
    Effectively this is similar to having ``class WFSController(FourierDualReference)`` inheritance.

    Since Micro-Manager / PyDevice does not yet support buttons to activate actions, a WFS experiment is started by setting
    the trigger attribute :attr:`wavefront` to the value State.OPTIMIZED
    It adds attributes for inspecting the statistics of the last WFS optimization.
    Manages the state of the wavefront and executes the algorithm to optimize and apply wavefront corrections, while
    exposing all these parameters to MicroManager.
    """

    class State(Enum):
        FLAT = 0
        OPTIMIZED = 1
        REOPTIMIZE = 2

    def __init__(self, _algorithm_class, *args, **kwargs):
        """
        Args:
            algorithm: An instance of a wavefront shaping algorithm.
        """
        super().__init__(*args, **kwargs)
        self._wavefront = WFSController.State.FLAT
        self._result: Optional[WFSResult] = None
        self._feedback_ratio = 0.0
        self._test_wavefront = False  # Trigger to test the optimized wavefront

    def __new__(cls, algorithm_class, *args, **kwargs):
        """Dynamically creates a class of type `class X(WFSController, algorithm_class` and returns an instance of that class"""

        # Dynamically create the class using type()
        class_name = "WFSController_" + algorithm_class.__name__
        DynamicClass = type(class_name, (cls, algorithm_class), {})
        instance = super(WFSController, cls).__new__(DynamicClass)
        return instance

    @property
    def wavefront(self) -> State:
        """
        Enables switching between FLAT or OPTIMIZED wavefront on the SLM.
        Setting this state to OPTIMIZED causes the algorithm execute if the optimized wavefront is not yet computed.
        Setting this state to REOPTIMIZE always causes the algorithm to recompute the wavefront. The state switches to OPTIMIZED after executioin of the algorithm.
        For multi-target optimizations, OPTIMIZED shows the wavefront for the first target.
        """
        return self._wavefront

    @wavefront.setter
    def wavefront(self, value):
        self._wavefront = WFSController.State(value)
        if value == WFSController.State.FLAT:
            self.slm.set_phases(0.0)
        elif value == WFSController.State.OPTIMIZED:
            if self._result is None:
                # run the algorithm
                self._result = self.execute().select_target(0)
            self.slm.set_phases(self.optimized_wavefront)
        else:  # value == WFSController.State.REOPTIMIZE:
            self._result = None  # remove stored result
            self.wavefront = WFSController.State.OPTIMIZED  # recompute the wavefront

    @property
    def fidelity_noise(self) -> float:
        """
        Returns:
            float: the estimated loss in fidelity caused by the limited snr.
        """
        return self._result.fidelity_noise if self._result is not None else 0.0

    @property
    def fidelity_amplitude(self) -> float:
        """
        Returns:
            float: estimated reduction of the fidelity due to phase-only
            modulation (≈ π/4 for fully developed speckle)
        """
        return self._result.fidelity_amplitude if self._result is not None else 0.0

    @property
    def estimated_enhancement(self) -> float:
        """
        Returns:
            float: estimated enhancement: estimated ratio <after>/<before>  (with <> denoting
            ensemble average)
        """
        return self._result.estimated_enhancement if self._result is not None else 0.0

    @property
    def fidelity_calibration(self) -> float:
        """
        Returns:
            float: non-linearity.
        """
        return self._result.fidelity_calibration if self._result is not None else 0.0

    @property
    def estimated_optimized_intensity(self) -> float:
        """
        Returns:
            float: estimated optimized intensity.
        """
        return self._result.estimated_optimized_intensity.mean() if self._result is not None else 0.0

    @property
    def snr(self) -> float:
        """
        Returns:
            float: The average signal-to-noise ratio (SNR) of the wavefront optimization measurements.
        """
        return self._result.snr if self._result is not None else 0.0

    @property
    def optimized_wavefront(self) -> np.ndarray:
        return -np.angle(self._result.t) if self._result is not None else 0.0

    @property
    def feedback_ratio(self) -> float:
        """The ratio of average feedback signals after and before optimization.

        This value is calculated when the :attr:`test_wavefront` trigger is set to True.

        Note: this is *not* the enhancement factor, because the 'before' signal is not ensemble averaged.
            Therefore, this value should be used with caution.

        Returns:
            float: average enhancement of the feedback, 0.0 none if no such enhancement was measured."""
        return self._feedback_ratio

    @property
    def test_wavefront(self) -> bool:
        """Trigger to test the wavefront.

        Set this value `True` to measure feedback signals with a flat and an optimized wavefront and compute the :attr:`feedback_ratio`.
        This value is reset to `False` after the test is performed.
        """
        return False

    @test_wavefront.setter
    def test_wavefront(self, value):
        if value:
            self.wavefront = WFSController.State.FLAT
            feedback_flat = self.feedback.read().sum()
            self.wavefront = WFSController.State.OPTIMIZED
            feedback_shaped = self.feedback.read().sum()
            self._feedback_ratio = float(feedback_shaped / feedback_flat)


class DummyProgressBar:
    """Placeholder for a progress bar object.

    Some functions take an optional tdqm-style progress bar as input.
    This class serves as a placeholder iif no progress bar is given.
    It does nothing.
    """

    def __init__(self):
        self.count = 0

    def update(self):
        pass
