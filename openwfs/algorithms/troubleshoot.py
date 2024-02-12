import time
from collections.abc import Callable

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt

from ..algorithms.utilities import WFSResult
from ..core import Detector


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
        Standard deviation of the signal, corrected for the variance due to given noise.
    """
    return float(np.sqrt(signal_with_noise.var() - noise.var()))


def cnr(signal_with_noise: np.ndarray, noise: np.ndarray) -> float:
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
    return float(signal_std(signal_with_noise, noise) / noise.std())


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
    return float(signal_std(signal_with_noise, noise) / signal_std(reference_with_noise, noise))


def cross_corr_fft2(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation with a 2D Fast Fourier Transform. Note that this approach will introduce wrap-around
    artefacts.

    Args:
        f, g:
            2D arrays to be correlated.
    """
    return ifft2(fft2(f).conj() * fft2(g))


def find_pixel_shift(f: np.ndarray, g: np.ndarray) -> tuple[np.intp, ...]:
    """
    Find the pixel shift between two images by performing a 2D FFT based cross-correlation.
    """
    corr = cross_corr_fft2(f, g)  # Compute cross-correlation with fft2
    s = np.array(corr).shape  # Get shape
    index = np.unravel_index(np.argmax(corr), s)  # Find 2D indices of maximum
    pix_shift = (fftfreq(s[0], 1 / s[0])[index[0]],  # Correct negative pixel shifts
                 fftfreq(s[1], 1 / s[1])[index[1]])
    return pix_shift


def field_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute field correlation, i.e. inner product of two fields, normalized by the product of the L2 norms,
    such that field_correlation(f, s*f) == 1, where s is a scalar value.
    Also known as normalized first order correlation :math:`g_1`.

    Args:
        A, B    Real or complex fields (or other arrays) to be correlated.
                A and B must have the same shape.
    """
    return np.vdot(A, B) / np.sqrt(np.vdot(A, A) * np.vdot(B, B))


def frame_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute frame correlation between two frames.
    This is the normalized second order correlation :math:`g_2`.
    Note: This function computes the correlation between two frames. The output is a single value.

    Args:
        A, B    Real-valued intensity arrays of the (such as camera frames) to be correlated.
                A and B must have the same shape.

    This correlation function can be used to estimate the lowered fidelity due to speckle decorrelation.
    See also:
         [Jang et al. 2015] https://opg.optica.org/boe/fulltext.cfm?uri=boe-6-1-72&id=306198
    """
    return np.mean(A * B) / (np.mean(A)*np.mean(B)) - 1


class StabilityResult:
    """
    Result of a stability measurement.
    """
    def __init__(self, pixel_shifts, correlations, contrast_ratios, timestamps):
        self.pixel_shifts = pixel_shifts
        self.correlations = correlations
        self.contrast_ratios = contrast_ratios
        self.timestamps = timestamps
        self.relative_timestamps = timestamps - timestamps[0]

    def plot(self):
        plt.plot(self.relative_timestamps, self.pixel_shifts, '.-', label='image-shift (pix)')
        plt.plot(self.relative_timestamps, self.contrast_ratios, '.-', label='contrast ratio')
        plt.plot(self.relative_timestamps, self.correlations, '.-', label='correlation')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()


def measure_setup_stability(frame_source, sleep_time_s, num_of_frames, dark_frame) -> StabilityResult:
    """Test the setup stability by repeatedly reading frames."""
    first_frame = frame_source.read()
    pixel_shifts = np.zeros(shape=(num_of_frames, 2))
    correlations = np.zeros(shape=(num_of_frames,))
    contrast_ratios = np.zeros(shape=(num_of_frames,))
    timestamps = np.zeros(shape=(num_of_frames,))

    # Start capturing frames
    for n in range(num_of_frames):
        time.sleep(sleep_time_s)
        new_frame = frame_source.read()
        pixel_shifts[n, :] = find_pixel_shift(first_frame, new_frame)
        correlations[n] = frame_correlation(first_frame, new_frame)
        contrast_ratios[n] = contrast_enhancement(new_frame, first_frame, dark_frame)
        timestamps[n] = time.perf_counter()

    return StabilityResult(pixel_shifts=pixel_shifts,
                           correlations=correlations,
                           contrast_ratios=contrast_ratios,
                           timestamps=timestamps)


def analyze_phase_calibration(wfs_result: WFSResult) -> float:
    r"""
    Analyze phase calibration.
    Estimate the fidelity reduction factor due to wrong phase calibration
    :math:`\langle|\gamma_\phi|^2\rangle = |c_1|^2 / \sum_{k\neq0} |c_k|^2`,
    where :math:`c_k` are components of the Fourier transform of phase stepping measurements. For perfectly calibrated
    SLMs, :math:`\langle|\gamma_\phi|^2\rangle` should converge to 1.

    Args:
        wfs_result: The WFSResult object containing the Fourier transform of the phase stepping measurements.
            Minimum number of phase steps=5, but a significantly higher number of phase steps is recommended for more
            accurate estimation.

    Returns:
        Estimate of fidelity reduction due to wrong phase calibration.
    """
    F = wfs_result.t_f  # Fourier transform of phase stepping measurements

    # Prepare indexing
    axis_k = wfs_result.axis  # Axis of phase stepping
    axis_not_k = list(range(F.ndim))  # List of all axis indices
    axis_not_k.pop(axis_k)  # Remove axis of phase stepping
    k_nyquist = int(np.floor((F.shape[axis_k] - 1) / 2))  # Nyquist frequency index

    # Compute components
    F1 = np.expand_dims(np.take(F, 1, axis=axis_k), axis=axis_k)  # Base frequency
    Fk = np.take(F, range(1, k_nyquist), axis=axis_k)  # All frequencies from base to Nyquist
    inner_sum = np.sum(Fk * F1.conj(), axis=tuple(axis_not_k), keepdims=True)  # Sum over all axes except phase stepping
    return np.sum((np.abs(F1)**2))**2 / np.sum(np.abs(inner_sum)**2, axis=axis_k)  # Compute |c1|²/∑|ck|²


class WFSTroubleshootResult:
    """
    Data structure for holding wavefront shaping statistics and additional troubleshooting information.

    Properties:
        fidelity_non_modulated: The estimated fidelity reduction factor due to the presence of non-modulated light.
        phase_calibration_ratio: A ratio indicating the correctness of the SLM phase response. An incorrect phase
            response produces a value < 1.
        wfs_result (WFSResult): Object containing the analyzed result of running the WFS algorithm.
        frame_signal_std_before: Signal unbiased standard deviation of frame captured before running the WFS algorithm.
        frame_signal_std_after: Signal unbiased standard deviation of frame captured after running the WFS algorithm.
        frame_cnr_before: Unbiased contrast ratio of frame captured before running the WFS algorithm.
        frame_cnr_after: Unbiased contrast ratio of frame captured after running the WFS algorithm.
        frame_contrast_enhancement: The unbiased contrast enhancement due to WFS. A frame taken with a flat wavefront is
            compared to a frame taken with a shaped wavefront. Both frames are taken after the WFS algorithm has run.
        frame_photobleaching_ratio: The signal degradation after WFS, typically caused by photo-bleaching in fluorescent
            experiments. A frame from before and after running the WFS are compared by computing the unbiased contrast
            enhancement factor. If this value is < 1, it means the signal has degraded by this factor.
        dark_frame: Frame taken with the laser blocked, before running the WFS algorithm.
        before_frame: Frame taken with the laser unblocked, before running the WFS algorithm, with a flat wavefront.
        after_frame: Frame taken with the laser unblocked, after running the WFS algorithm, with a flat wavefront.
        shaped_frame: Frame taken with the laser unblocked, after running the WFS algorithm, with a shaped wavefront.
        stability (StabilityResult): Object containing the result of the stability test.
    """
    def __init__(self):
        # Fidelities and WFS metrics
        self.fidelity_non_modulated = None
        self.phase_calibration_ratio = None

        # WFS result
        self.wfs_result = None

        # Frame metrics
        self.frame_signal_std_before = None
        self.frame_signal_std_after = None
        self.frame_cnr_before = None
        self.frame_cnr_after = None
        self.frame_contrast_enhancement = None
        self.frame_photobleaching_ratio = None
        self.stability = None

        # Frames
        self.dark_frame = None
        self.before_frame = None
        self.after_frame = None
        self.shaped_frame = None

        # Other
        self.timestamp = time.time()

    def report(self):
        """
        Print a report of all results to the console.
        """
        print(f'\n===========================')
        print(f'{time.ctime(self.timestamp)}\n')
        print(f'=== Measurement metrics ===')
        print(f'fidelity_non_modulated = {self.fidelity_non_modulated}:.3f')
        print(f'fidelity_phase_calibration = {self.phase_calibration_ratio}:.3f')
        print(f'')
        print(f'=== Frame metrics ===')
        print(f'σ_signal, before = {self.frame_signal_std_before:.3f}')
        print(f'σ_signal, after = {self.frame_signal_std_after:.3f}')
        print(f'CNR before = {self.frame_cnr_before:.3f}')
        print(f'CNR after = {self.frame_cnr_after:.3f}')
        print(f'Contrast enhancement η_σ = {self.frame_contrast_enhancement:.3f}')
        print(f'Photobleaching ratio = {self.frame_photobleaching_ratio:.3f}')

        if self.stability is not None:
            self.stability.plot()


def troubleshoot(algorithm, frame_source: Detector,
                 laser_unblock: Callable, laser_block: Callable,
                 do_frame_capture=True, do_stability_test=True, do_log=True,
                 stability_sleep_time_s=0.5,
                 stability_num_of_frames=500) -> WFSTroubleshootResult:
    """
    Run a series of basic checks to find common sources of error in a WFS experiment.
    Quantifies several types of fidelity reduction.

    Args:
        algorithm: Wavefront Shaping algorithm object, e.g. StepwiseSequential.
        frame_source: Source object for reading frames, e.g. Camera.
        laser_unblock: Function to run for unblocking the laser light.
        laser_block: Function to run for blocking the laser light.
        do_frame_capture: Boolean. If False, skip frame capture before and after running the WFS algorithm.
            Also skips computation of corresponding metrics. Also skips stability test.
        do_stability_test: Boolean. If False, skip stability test.
        do_log: Report what the troubleshooter is doing on the console.
        stability_sleep_time_s: Float. Sleep time in seconds in between capturing frames.
        stability_num_of_frames: Integer. Number of frames to take in the stability test.


    Returns:
        trouble: WFSTroubleshootResult object containing troubleshoot information.
    """

    # Initialize an empty WFS troubleshoot result
    trouble = WFSTroubleshootResult()

    if do_frame_capture:
        if do_log: print('Capturing frames before WFS...')

        # Capture frames before WFS
        algorithm.slm.set_phases(0.0)                                       # Flat wavefront
        laser_block()
        trouble.dark_frame = frame_source.read()                            # Dark frame
        laser_unblock()
        trouble.before_frame = frame_source.read()                          # Before frame (flat wf)

        # Frame metrics
        trouble.frame_signal_std_before = signal_std(trouble.before_frame, trouble.dark_frame)
        trouble.frame_cnr_before = cnr(trouble.before_frame, trouble.dark_frame)

    # WFS experiment
    if do_log: print('Run WFS algorithm...')
    trouble.wfs_result = algorithm.execute()                                # Execute WFS algorithm

    if do_frame_capture:
        if do_log: print('Capturing frames after WFS...')

        # Capture frames after WFS
        algorithm.slm.set_phases(0.0)                                       # Flat wavefront
        trouble.after_frame = frame_source.read()                           # After frame (flat wf)
        algorithm.slm.set_phases(-np.angle(trouble.wfs_result.t))           # Shaped wavefront
        trouble.shaped_wf_frame = frame_source.read()                       # Shaped wavefront frame

        # Frame metrics
        if do_log: print('Compute frame metrics...')
        trouble.frame_signal_std_after = signal_std(trouble.before_frame, trouble.dark_frame)
        trouble.frame_cnr_after = cnr(trouble.after_frame, trouble.dark_frame)  # Frame CNR after
        trouble.frame_contrast_enhancement = \
            contrast_enhancement(trouble.shaped_wf_frame, trouble.after_frame, trouble.dark_frame)
        trouble.frame_photobleaching_ratio = \
            contrast_enhancement(trouble.after_frame, trouble.before_frame, trouble.dark_frame)

    if do_stability_test and do_frame_capture:
        if do_log: print('Run stability test...')

        # Test setup stability
        trouble.stability = measure_setup_stability(
            frame_source=frame_source,
            sleep_time_s=stability_sleep_time_s,
            num_of_frames=stability_num_of_frames,
            dark_frame=trouble.dark_frame)

    # Analyze the WFS result
    if do_log: print('Analyze phase calibration...')
    trouble.fidelity_phase_calibration = analyze_phase_calibration(trouble.wfs_result)

    return trouble
