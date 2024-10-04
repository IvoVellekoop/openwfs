import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

from ..core import Detector, PhaseSLM


# TODO: review, replace by numpy/scipy functions where possible, remove or hide functions that are too specific
def signal_std(signal_with_noise: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute noise corrected standard deviation of signal measurement.

    Args:
        signal_with_noise:
            ND array containing the measured signal including noise. The noise is assumed to be uncorrelated with the
            signal, such that var(measured) = var(signal) + var(noise).
        noise:
            ND array containing only noise.

    Returns:
        Standard deviation of the signal, corrected for the variance due to given noise. If the noise variance
        is greater than the signal_with_noise variance, the output will be 0.
    """
    signal_var = signal_with_noise.var() - noise.var()
    if signal_var < 0:
        signal_var = 0
    return np.sqrt(signal_var)


def cnr(signal_with_noise: np.ndarray, noise: np.ndarray) -> np.float64:
    """
    Compute the noise-corrected contrast-to-noise ratio of a measured signal. Contrast is computed as the standard
    deviation, corrected for noise. The noise variance is computed from a separate array, containing only noise.

    Args:
        signal_with_noise:
            ND array containing the measured signal including noise. The noise is assumed to be uncorrelated with the
            signal, such that var(measured) = var(signal) + var(noise).
        noise:
            ND array containing only noise, e.g. a dark frame.

    Returns:
        Standard deviation of the signal, corrected for the variance due to given noise.
    """
    return signal_std(signal_with_noise, noise) / noise.std()


def contrast_enhancement(signal_with_noise: np.ndarray, reference_with_noise: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute noise corrected contrast enhancement. The noise is assumed to be uncorrelated with the signal, such that
    var(measured) = var(signal) + var(noise).

    Args:
        signal_with_noise:
            ND array containing the measured signal including noise, e.g. image signal with shaped wavefront.
        reference_with_noise:
            ND array containing a reference signal including noise, e.g. image signal with a flat wavefront.
        noise:
            ND array containing only noise.

    Returns:
        Standard deviation of the signal, corrected for the variance due to given noise.
    """
    signal_std_sig = signal_std(signal_with_noise, noise)
    signal_std_ref = signal_std(reference_with_noise, noise)
    return signal_std_sig / signal_std_ref


def cross_corr_fft2(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation with a 2D Fast Fourier Transform. Note that this approach will introduce wrap-around
    artefacts.

    Args:
        f, g:
            2D arrays to be correlated.
    """
    return ifft2(fft2(f).conj() * fft2(g))


def find_pixel_shift(f: np.ndarray, g: np.ndarray) -> tuple[int, ...]:
    """
    Find the pixel shift between two images by performing a 2D FFT based cross-correlation.
    """
    corr = cross_corr_fft2(f, g)  # Compute cross-correlation with fft2
    s = np.array(corr).shape  # Get shape
    index = np.unravel_index(np.argmax(corr), s)  # Find 2D indices of maximum
    # convert indices to pixel shift
    dy = index[0] if index[0] < s[0] / 2 else index[0] - s[0]
    dx = index[1] if index[1] < s[1] / 2 else index[1] - s[1]
    return dy, dx


def field_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute field correlation, i.e. inner product of two fields, normalized by the product of the L2 norms,
    such that field_correlation(f, s*f) == 1, where s is a scalar value.
    Also known as normalized first order correlation :math:`g_1`.

    Args:
        A, B    Real or complex fields (or other arrays) to be correlated.
                A and B must have the same shape.
    """
    return np.vdot(a, b) / np.sqrt(np.vdot(a, a) * np.vdot(b, b))


def frame_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute frame correlation between two frames.
    This is the normalized second order correlation :math:`g_2`.
    Note: This function computes the correlation between two frames. The output is a single value.

    Args:
        a, b    Real-valued intensity arrays of the (such as camera frames) to be correlated.
                A and B must have the same shape.

    This correlation function can be used to estimate the lowered fidelity due to speckle decorrelation.
    See also:
         [Jang et al. 2015] https://opg.optica.org/boe/fulltext.cfm?uri=boe-6-1-72&id=306198
    """
    return np.mean(a * b) / (np.mean(a) * np.mean(b)) - 1


def pearson_correlation(a: np.ndarray, b: np.ndarray, noise_var: np.ndarray = 0.0) -> float:
    """
    Compute Pearson correlation.

    The variances in the normalization factor can optionally be compensated for uncorrelated noise. This is done
    by subtracting the noise variance from the signal variance.

    Args:
        a: real-valued input array.
        b: real-valued input array.
        noise_var: Variance of uncorrelated noise to compensate for.
    """
    a_dev = a - a.mean()  # Deviations from mean a
    b_dev = b - b.mean()  # Deviations from mean b
    covar = (a_dev * b_dev).mean()  # Covariance
    a_var_signal = a.var() - noise_var  # Variance of signal in ``a``, excluding noise
    b_var_signal = b.var() - noise_var  # Variance of signal in ``b``, excluding noise
    return covar / np.sqrt(a_var_signal * b_var_signal)


class StabilityResult:
    """
    Result of a stability measurement.

    Attributes:
        pixel_shifts_first: Image shift in pixels, compared to first frame.
        correlations_first: Pearson correlations with first frame.
        correlations_disattenuated_first: Pearson correlations with first frame. Compensated for noise.
        contrast_ratios_first: Contrast ratio with first frame.
        pixel_shifts_prev: Image shift in pixels, compared to previous frame.
        correlations_prev: Pearson correlations with previous frame.
        correlations_disattenuated_prev: Pearson correlations with previous frame. Compensated for noise.
        contrast_ratios_prev: Contrast ratio with previous frame.
        timestamps: Timestamps in seconds since start of measurement.
        framestack: 3D array containing all recorded frames. Is None unless saving frames was requested.
    """

    def __init__(
        self,
        pixel_shifts_first,
        correlations_first,
        correlations_disattenuated_first,
        contrast_ratios_first,
        pixel_shifts_prev,
        correlations_prev,
        correlations_disattenuated_prev,
        contrast_ratios_prev,
        abs_timestamps,
        framestack,
    ):
        # Comparison with first frame
        self.pixel_shifts_first = pixel_shifts_first
        self.correlations_first = correlations_first
        self.correlations_disattenuated_first = correlations_disattenuated_first
        self.contrast_ratios_first = contrast_ratios_first

        # Comparison with previous frame
        self.pixel_shifts_prev = pixel_shifts_prev
        self.correlations_prev = correlations_prev
        self.correlations_disattenuated_prev = correlations_disattenuated_prev
        self.contrast_ratios_prev = contrast_ratios_prev

        # Other
        self.timestamps = abs_timestamps - abs_timestamps[0]
        self.framestack = framestack

    def plot(self):
        """
        Plot stability test results: image shift (x and y), correlation, contrast ratio,
        all with respect to first frame and previous frame.
        """
        # Comparisons with first frame
        plt.figure()
        plt.plot(self.timestamps, self.pixel_shifts_first, ".-", label="image-shift (pix)")
        plt.title("Stability - Image shift w.r.t. first frame")
        plt.ylabel("Image shift (pix)")
        plt.xlabel("time (s)")

        plt.figure()
        plt.plot(self.timestamps, self.correlations_first, ".-", label="correlation")
        plt.plot(
            self.timestamps,
            self.correlations_disattenuated_first,
            ".-",
            label="correlation disattenuated",
        )
        plt.title("Stability - Correlation with first frame")
        plt.xlabel("time (s)")
        plt.legend()

        plt.figure()
        plt.plot(self.timestamps, self.contrast_ratios_first, ".-", label="contrast ratio")
        plt.title("Stability - Contrast ratio with first frame")
        plt.xlabel("time (s)")

        # Comparisons with previous frame
        plt.figure()
        plt.plot(self.timestamps, self.pixel_shifts_prev, ".-", label="image-shift (pix)")
        plt.title("Stability - Image shift w.r.t. previous frame")
        plt.ylabel("Image shift (pix)")
        plt.xlabel("time (s)")

        plt.figure()
        plt.plot(self.timestamps, self.correlations_prev, ".-", label="correlation")
        plt.plot(
            self.timestamps,
            self.correlations_disattenuated_prev,
            ".-",
            label="correlation disattenuated",
        )
        plt.title("Stability - Correlation with previous frame")
        plt.xlabel("time (s)")
        plt.legend()

        plt.figure()
        plt.plot(self.timestamps, self.contrast_ratios_prev, ".-", label="contrast ratio")
        plt.title("Stability - Contrast ratio with previous frame")
        plt.xlabel("time (s)")

        plt.show()


def measure_setup_stability(
    frame_source, sleep_time_s, num_of_frames, dark_frame, do_save_frames=False
) -> StabilityResult:
    """Test the setup stability by repeatedly reading frames."""
    first_frame = frame_source.read()
    prev_frame = first_frame

    # Initialize arrays - first frame comparisons
    pixel_shifts_first = np.zeros(shape=(num_of_frames, 2))
    correlations_first = np.zeros(shape=(num_of_frames,))
    correlations_disattenuated_first = np.zeros(shape=(num_of_frames,))
    contrast_ratios_first = np.zeros(shape=(num_of_frames,))

    # Initialize arrays - previous frame comparisons
    pixel_shifts_prev = np.zeros(shape=(num_of_frames, 2))
    correlations_prev = np.zeros(shape=(num_of_frames,))
    correlations_disattenuated_prev = np.zeros(shape=(num_of_frames,))
    contrast_ratios_prev = np.zeros(shape=(num_of_frames,))
    abs_timestamps = np.zeros(shape=(num_of_frames,))

    if do_save_frames:
        framestack_shape = first_frame.shape + (num_of_frames,)
        framestack = np.zeros(shape=framestack_shape)
    else:
        framestack = None

    dark_var = dark_frame.var()

    # Start capturing frames
    for n in range(num_of_frames):
        time.sleep(sleep_time_s)
        new_frame = frame_source.read()

        # Compare with first frame
        pixel_shifts_first[n, :] = find_pixel_shift(first_frame, new_frame)
        correlations_first[n] = pearson_correlation(first_frame, new_frame)
        correlations_disattenuated_first[n] = pearson_correlation(first_frame, new_frame, noise_var=dark_var)
        contrast_ratios_first[n] = contrast_enhancement(new_frame, first_frame, dark_frame)

        # Compare with previous frame
        pixel_shifts_prev[n, :] = find_pixel_shift(prev_frame, new_frame)
        correlations_prev[n] = pearson_correlation(prev_frame, new_frame)
        correlations_disattenuated_prev[n] = pearson_correlation(prev_frame, new_frame, noise_var=dark_var)
        contrast_ratios_prev[n] = contrast_enhancement(new_frame, prev_frame, dark_frame)
        abs_timestamps[n] = time.perf_counter()

        # Save frame if requested
        if do_save_frames:
            framestack[:, :, n] = new_frame

        prev_frame = new_frame

    return StabilityResult(
        pixel_shifts_first=pixel_shifts_first,
        correlations_first=correlations_first,
        correlations_disattenuated_first=correlations_disattenuated_first,
        contrast_ratios_first=contrast_ratios_first,
        pixel_shifts_prev=pixel_shifts_prev,
        correlations_prev=correlations_prev,
        correlations_disattenuated_prev=correlations_disattenuated_prev,
        contrast_ratios_prev=contrast_ratios_prev,
        abs_timestamps=abs_timestamps,
        framestack=framestack,
    )


def measure_modulated_light_dual_phase_stepping(slm: PhaseSLM, feedback: Detector, phase_steps: int, num_blocks: int):
    """
    Measure the ratio of modulated light with the dual phase stepping method.

    Args:
        slm: The SLM that will be used to modulate the light.
        feedback: The Detector providing the feedback.
        phase_steps: Number of phase steps for each group. Must be >=3. Total number of measurements will be
            phase_steps².
        num_blocks: Number of blocks in each dimension of the checkerboard pattern that is used to create the
            modulated groups.

    Returns:
        An estimate of the ratio of modulated light. Independent of any uncorrelated intensity offset.

    Note: In the calculation of the ratio of modulated light, a small epsilon term is added to prevent division by zero.
    """
    assert phase_steps >= 3

    # Initialization
    check_lin = np.arange(num_blocks).reshape(num_blocks, 1)
    block_pattern_p = np.mod(check_lin + check_lin.T, 2)
    block_pattern_q = 1 - block_pattern_p
    measurements = np.zeros((phase_steps, phase_steps))

    # Dual phase stepping
    for p in range(phase_steps):
        phase_p = p * 2 * np.pi / phase_steps
        for q in range(phase_steps):
            phase_q = q * 2 * np.pi / phase_steps
            phase_pattern = block_pattern_p * phase_p + block_pattern_q * phase_q
            slm.set_phases(phase_pattern)
            measurements[p, q] = feedback.read()

    # 2D Fourier transform the modulation measurements
    f = np.fft.fft2(measurements) / phase_steps**2

    # Compute fidelity factor due to modulated light
    eps = 1e-6  # Epsilon term to prevent division by zero
    m1_m2_ratio = (np.abs(f[0, 1]) ** 2 + eps) / (np.abs(f[1, 0]) ** 2 + eps)  # Ratio of modulated intensities
    fidelity_modulated = (1 + m1_m2_ratio) / (1 + m1_m2_ratio + np.abs(f[0, 1]) ** 2 / np.abs(f[1, -1]) ** 2)

    return fidelity_modulated


def measure_modulated_light(slm: PhaseSLM, feedback: Detector, phase_steps: int):
    """
    Measure the ratio of modulated light by phase stepping the entire SLM.

    Args:
        slm: The SLM that will be used to modulate the light.
        feedback: The Detector providing the feedback.
        phase_steps: Number of phase steps. Must be >=3.

    Returns:
        fidelity_modulated: An estimate of the ratio of modulated light.

    Note: It is not possible to detect which field is modulated and which is static. This calculation assumes
        the modulated intensity is greater than the non-modulated intensity.
    """
    assert phase_steps >= 3

    # Initialization
    measurements = np.zeros((phase_steps,))

    # Dual phase stepping
    for p in range(phase_steps):
        slm.set_phases(p * 2 * np.pi / phase_steps)
        measurements[p] = feedback.read()

    # 2D Fourier transform the modulation measurements
    f = np.fft.fft(measurements)

    # Compute ratio of modulated light over total
    fidelity_modulated = 0.5 * (1.0 + np.sqrt(np.clip(1.0 - 4.0 * np.abs(f[1] / f[0]) ** 2, 0, None)))

    return fidelity_modulated


class WFSTroubleshootResult:
    """
    Data structure for holding wavefront shaping statistics and additional troubleshooting information.

    Attributes:
        fidelity_non_modulated: The estimated fidelity reduction factor due to the presence of non-modulated light.
        fidelity_phase_calibration: A ratio indicating the correctness of the SLM phase response. An incorrect phase
            response produces a value < 1.
        wfs_result (WFSResult): Object containing the analyzed result of running the WFS algorithm.
        feedback_before: Feedback from before running the WFS algorithm, with a flat wavefront.
        feedback_after: Feedback from after running the WFS algorithm, with a flat wavefront.
        frame_signal_std_before: Signal unbiased standard deviation of frame captured before running the WFS algorithm.
        frame_signal_std_after: Signal unbiased standard deviation of frame captured after running the WFS algorithm.
        frame_cnr_before: Unbiased contrast ratio of frame captured before running the WFS algorithm.
        frame_cnr_after: Unbiased contrast ratio of frame captured after running the WFS algorithm.
        frame_contrast_enhancement: The unbiased contrast enhancement due to WFS. A frame taken with a flat wavefront is
            compared to a frame taken with a shaped wavefront. Both frames are taken after the WFS algorithm has run.
        frame_photobleaching_ratio: The signal degradation after WFS, typically caused by photo-bleaching in fluorescent
            experiments. A frame from before and after running the WFS are compared by computing the unbiased contrast
            enhancement factor. If this value is < 1, it means the signal has degraded by this factor.
        frame_repeatability: Pearson correlation of two frames taken before running the WFS algorithm,
            with a flat wavefront. Not compensated for noise.
        dark_frame: Frame taken with the laser blocked, before running the WFS algorithm.
        before_frame: Frame taken with the laser unblocked, before running the WFS algorithm, with a flat wavefront.
        after_frame: Frame taken with the laser unblocked, after running the WFS algorithm, with a flat wavefront.
        shaped_wf_frame: Frame taken with the laser unblocked, after running the WFS algorithm, with a shaped wavefront.
        stability (StabilityResult): Object containing the result of the stability test.
    """

    def __init__(self):
        # Fidelity and WFS metrics
        self.fidelity_non_modulated = None
        self.fidelity_phase_calibration = None
        self.fidelity_decorrelation = None
        self.expected_enhancement = None

        # WFS
        self.wfs_result = None
        self.average_background = None
        self.feedback_before = None
        self.feedback_after = None
        self.measured_enhancement = None

        # Frame metrics
        self.frame_signal_std_before = None
        self.frame_signal_std_after = None
        self.frame_signal_std_shaped_wf = None
        self.frame_cnr_before = None
        self.frame_cnr_after = None
        self.frame_cnr_shaped_wf = None
        self.frame_contrast_enhancement = None
        self.frame_photobleaching_ratio = None
        self.frame_repeatability = None
        self.stability = None

        # Frames
        self.dark_frame = None
        self.before_frame = None
        self.after_frame = None
        self.shaped_wf_frame = None

        # Other
        self.timestamp = time.time()

    def report(self, do_plots=True):
        """
        Print a report of all results to the console.

        Args:
            do_plots (bool): Plot some results as graphs.
        """
        print(f"\n===========================")
        print(f"{time.ctime(self.timestamp)}\n")
        print(f"=== Feedback metrics ===")
        print(f"number of modes (N): {self.wfs_result.n:.3f}")
        print(f"fidelity_amplitude: {self.wfs_result.fidelity_amplitude.squeeze():.3f}")
        print(f"fidelity_noise: {self.wfs_result.fidelity_noise.squeeze():.3f}")
        print(f"fidelity_non_modulated: {self.fidelity_non_modulated:.3f}")
        print(f"fidelity_phase_calibration: {self.wfs_result.fidelity_calibration.squeeze():.3f}")
        print(f"fidelity_decorrelation: {self.fidelity_decorrelation:.3f}")
        print(f"expected enhancement: {self.expected_enhancement:.3f}")
        print(f"measured enhancement: {self.measured_enhancement:.3f}")
        print(f"")
        print(f"=== Frame metrics ===")
        print(f"signal std, before: {self.frame_signal_std_before:.2f}")
        print(f"signal std, after: {self.frame_signal_std_after:.2f}")
        print(f"signal std, with shaped wavefront: {self.frame_signal_std_shaped_wf:.2f}")
        if self.dark_frame is not None:
            print(f"average offset (dark frame): {self.dark_frame.mean():.2f}")
            print(f"median offset (dark frame): {np.median(self.dark_frame):.2f}")
            print(f"noise std (dark frame): {np.std(self.dark_frame):.2f}")
        print(f"frame repeatability: {self.frame_repeatability:.3f}")
        print(f"contrast to noise ratio before: {self.frame_cnr_before:.3f}")
        print(f"contrast to noise ratio after: {self.frame_cnr_after:.3f}")
        print(f"contrast to noise ratio with shaped wavefront: {self.frame_cnr_shaped_wf:.3f}")
        print(f"contrast enhancement: {self.frame_contrast_enhancement:.3f}")
        print(f"photobleaching ratio: {self.frame_photobleaching_ratio:.3f}")

        if do_plots and self.stability is not None:
            self.stability.plot()

        if (
            do_plots
            and self.dark_frame is not None
            and self.after_frame is not None
            and self.shaped_wf_frame is not None
        ):
            max_value = max(self.after_frame.max(), self.shaped_wf_frame.max())

            # Plot dark frame
            plt.figure()
            plt.imshow(self.dark_frame, vmin=0, vmax=max_value)
            plt.title("Dark frame")
            plt.colorbar()
            plt.xlabel("x (pix)")
            plt.ylabel("y (pix)")
            plt.figure()

            # Plot after frame with flat wf
            plt.imshow(self.after_frame, vmin=0, vmax=max_value)
            plt.title("Frame with flat wavefront")
            plt.colorbar()
            plt.xlabel("x (pix)")
            plt.ylabel("y (pix)")

            # Plot shaped wf frame
            plt.figure()
            plt.imshow(self.shaped_wf_frame, vmin=0, vmax=max_value)
            plt.title("Frame with shaped wavefront")
            plt.colorbar()
            plt.xlabel("x (pix)")
            plt.ylabel("y (pix)")
            plt.show()


def troubleshoot(
    algorithm,
    background_feedback: Detector,
    frame_source: Detector,
    shutter,
    do_frame_capture=True,
    do_long_stability_test=False,
    stability_sleep_time_s=0.5,
    stability_num_of_frames=500,
    stability_do_save_frames=False,
    measure_non_modulated_phase_steps=16,
) -> WFSTroubleshootResult:
    """
    Run a series of basic checks to find common sources of error in a WFS experiment.
    Quantifies several types of fidelity reduction.

    Args:
        measure_non_modulated_phase_steps:
        algorithm: Wavefront Shaping algorithm object, e.g. StepwiseSequential.
        background_feedback: Feedback source that determines average background speckle intensity.
        frame_source: Source object for reading frames, e.g. Camera.
        shutter: Device object that can block/unblock light source.
        do_frame_capture: Boolean. If False, skip frame capture before and after running the WFS algorithm.
            Also skips computation of corresponding metrics. Also skips stability test.
        do_long_stability_test: Boolean. If False, skip long stability test where many frames are captured over
            a longer period of time.
        stability_sleep_time_s: Float. Sleep time in seconds in between capturing frames for long stability test.
        stability_num_of_frames: Integer. Number of frames to take in long stability test.
        stability_do_save_frames: Boolean. If True, save all recorded frames.
        measure_non_modulated_phase_steps: Integer. Number of phase steps for determining non-modulated light.


    Returns:
        trouble: WFSTroubleshootResult object containing troubleshoot information.
    """

    # Initialize an empty WFS troubleshoot result
    trouble = WFSTroubleshootResult()

    if do_frame_capture:
        logging.info("Capturing frames before WFS...")

        # Capture frames before WFS
        algorithm.slm.set_phases(0.0)  # Flat wavefront
        shutter.open = False
        trouble.dark_frame = frame_source.read()  # Dark frame
        shutter.open = True
        trouble.before_frame = frame_source.read()  # Before frame (flat wf)
        before_frame_2 = frame_source.read()

        # Frame metrics
        trouble.frame_signal_std_before = signal_std(trouble.before_frame, trouble.dark_frame)
        trouble.frame_cnr_before = cnr(trouble.before_frame, trouble.dark_frame)
        trouble.frame_repeatability = pearson_correlation(trouble.before_frame, before_frame_2)

    if do_long_stability_test and do_frame_capture:
        logging.info("Run long stability test...")

        # Test setup stability
        trouble.stability = measure_setup_stability(
            frame_source=frame_source,
            sleep_time_s=stability_sleep_time_s,
            num_of_frames=stability_num_of_frames,
            dark_frame=trouble.dark_frame,
            do_save_frames=stability_do_save_frames,
        )

    trouble.feedback_before = algorithm.feedback.read()

    # WFS experiment
    logging.info("Run WFS algorithm...")
    trouble.wfs_result = algorithm.execute()  # Execute WFS algorithm

    # Flat wavefront
    algorithm.slm.set_phases(0.0)
    trouble.average_background = background_feedback.read()
    trouble.feedback_after = algorithm.feedback.read()

    if do_frame_capture:
        logging.info("Capturing frames after WFS...")
        trouble.after_frame = frame_source.read()  # After frame (flat wf)

    # Shaped wavefront
    algorithm.slm.set_phases(-np.angle(trouble.wfs_result.t))
    trouble.feedback_shaped_wf = algorithm.feedback.read()

    trouble.measured_enhancement = trouble.feedback_shaped_wf / trouble.average_background

    if do_frame_capture:
        trouble.shaped_wf_frame = frame_source.read()  # Shaped wavefront frame

        # Frame metrics
        logging.info("Compute frame metrics...")
        trouble.frame_signal_std_after = signal_std(trouble.after_frame, trouble.dark_frame)
        trouble.frame_signal_std_shaped_wf = signal_std(trouble.shaped_wf_frame, trouble.dark_frame)
        trouble.frame_cnr_after = cnr(trouble.after_frame, trouble.dark_frame)  # Frame CNR after
        trouble.frame_cnr_shaped_wf = cnr(trouble.shaped_wf_frame, trouble.dark_frame)  # Frame CNR shaped wf
        trouble.frame_contrast_enhancement = contrast_enhancement(
            trouble.shaped_wf_frame, trouble.after_frame, trouble.dark_frame
        )
        trouble.frame_photobleaching_ratio = contrast_enhancement(
            trouble.after_frame, trouble.before_frame, trouble.dark_frame
        )
        trouble.fidelity_decorrelation = pearson_correlation(
            trouble.before_frame,
            trouble.after_frame,
            noise_var=trouble.dark_frame.var(),
        )

    trouble.fidelity_non_modulated = measure_modulated_light(
        slm=algorithm.slm,
        feedback=algorithm.feedback,
        phase_steps=measure_non_modulated_phase_steps,
    )

    trouble.expected_enhancement = np.squeeze(
        trouble.wfs_result.n
        * trouble.wfs_result.fidelity_amplitude
        * trouble.wfs_result.fidelity_noise
        * trouble.fidelity_non_modulated
        * trouble.wfs_result.fidelity_calibration
        * trouble.fidelity_decorrelation
    )

    # Analyze the WFS result
    logging.info("Analyze WFS result...")
    return trouble
