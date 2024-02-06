import time
from collections.abc import Callable

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

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


def measure_setup_stability(frame_source, wait_time_s, num_of_frames):
    """Test the setup stability by repeatedly reading frames."""
    first_frame = frame_source.read()
    pixel_shifts = np.zeros(shape=(num_of_frames, 2))
    correlations = np.zeros(shape=(num_of_frames,))

    for n in range(num_of_frames):
        # self.set_slm_random()
        time.sleep(wait_time_s)
        # self._wavefront = self.State.FLAT_WAVEFRONT

        new_frame = frame_source.read()
        pixel_shifts[n, :] = find_pixel_shift(first_frame, new_frame)
        correlations[n] = frame_correlation(first_frame, new_frame)

    return pixel_shifts, correlations


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
    """
    def __init__(self):
        # Fidelities
        self.fidelity_non_modulated = None
        self.fidelity_phase_calibration = None

        # Frames
        self.dark_frame = None
        self.before_frame = None
        self.after_frame = None
        self.shaped_frame = None

        # Metrics
        self.frame_cnr = None
        self.frame_signal_std = None
        self.contrast_enhancement = None
        self.stability = None
        self.photobleaching = None
        self.slm_timing = None

        # WFS result
        self.wfs_result = None


def troubleshoot(algorithm, frame_source: Detector,
                 laser_unblock: Callable, laser_block: Callable) -> WFSTroubleshootResult:
    """
    Run a series of basic checks to find common sources of error in a WFS experiment.
    Quantifies several types of fidelity reduction.

    Args:
        algorithm: Wavefront Shaping algorithm object, e.g. StepwiseSequential.
        frame_source: Source object for reading frames, e.g. Camera.
        laser_unblock: Function to run for unblocking the laser light.
        laser_block: Function to run for blocking the laser light.

    Returns:
        trouble: WFSTroubleshootResult object containing troubleshoot information.
    """

    # Initialize an empty WFS troubleshoot result
    trouble = WFSTroubleshootResult()

    # Capture frames before WFS
    algorithm.slm.set_phases(0.0)  # Flat wavefront
    laser_block()
    trouble.dark_frame = frame_source.read()  # Dark frame
    laser_unblock()
    trouble.before_frame = frame_source.read()  # Before frame (flat wf)

    # Frame CNR
    trouble.frame_cnr = cnr(trouble.before_frame, trouble.dark_frame)

    # WFS experiment
    trouble.wfs_result = algorithm.execute().select_target(0)  # Execute WFS algorithm

    # Capture frames after WFS
    algorithm.slm.set_phases(0.0)  # Flat wavefront
    trouble.after_frame = frame_source.read()  # After frame (flat wf)
    algorithm.slm.set_phases(-np.angle(trouble.wfs_result.t))  # Shaped wavefront
    trouble.shaped_wf_frame = frame_source.read()  # Shaped wavefront frame

    # Contrast enhancement
    trouble.contrast_enhancement = contrast_enhancement(trouble.shaped_wf_frame, trouble.after_frame, trouble.dark_frame)

    trouble.fidelity_phase_calibration = analyze_phase_calibration(trouble.wfs_result)

    # Test setup stability
    pixel_shifts, correlations = measure_setup_stability()

    ### Debugging
    import matplotlib.pyplot as plt
    plt.plot(pixel_shifts[:, 0], label='x')
    plt.plot(pixel_shifts[:, 1], label='y')
    plt.plot(correlations, label='Corr. with first')
    plt.legend()
    plt.show()

    return trouble


    ### === Test setup stability ===
    ### Requirement: find-image-pixel-shift function
    ### Preconfig: Flat wavefront
    ### Measurement: Snap multiple images over a long period of time
    ### Calculation: Cross-correlation between images
    ### Result: xdrift, ydrift, intensity drift, warning if >threshold
    ### Note 1: measurement should take a long time, could trouble in significant photobleaching
    ### Note 2: larger image size for more precise x,y cross correlation
    ### Implementation complexity: 3

    ### === Check SLM timing ===
    ### Measurement: Quick measurement, change SLM pattern, quick measurement, wait,
    ### later measurement
    ### Calculation: do quick measurement and later measurement correspond
    ### Result 1: Warning if timing seems incorrect
    ### Result 2: Plot graph
    ### Implementation complexity: 5

    ### === SLM illumination ===
    ### Requirement: Depends on it's own experiment
    ### Measurement: WFS experiment on entire SLM
    ### Calculation: amplitude from WFS (from e.g. Hadamard to SLM x,y basis)
    ### Result: SLM illumination map
    ### Implementation complexity: 4

    ### === Measure unmodulated light ===
    ### Requirement: Depends on it's own experiment
    ### Measurement: 2-mode phase stepping checkerboard.
    ### Calculation:
    ###    |A⋅exp(iθ) + B⋅exp(iφ) + C|² + b.g.noise
    ### Result: fraction of modulated and fraction of unmodulated light, warning if >threshold
    ### Note: large fraction of unmodulated light could indicate wrong illumination
    ### Implementation complexity: 7

    ### === Check calibration LUT ===
    ### Requirement: Phase stepping measurement light modulation done
    ### Calculation: Check higher phasestep frequencies. Is response cosine?
    ### Result 1: Nonlinearity value. Warning if very not cosine
    ### Result 2: Plot graph
    ### Implementation complexity: 5

    ### === Done: Quantify CNR ===
    ### Requirement: darkframe, before frame, noise corrected std function
    ### Calculation: noise corrected std / std of darkframe
    ### Result: 'measured signal when dark is noise'-SNR

    ### === Quantify SNR - with frame correlation ===
    ### Requirement: cross-corr
    ### Measurement: Snap multiple images in short time
    ### Calculation: how do consecutive frames correlate?
    ### Result: 'everything not reproducible is noise'-SNR
    ### Note: How can we distinguish from slm phase jitter?
    ### Implementation complexity: 2

    ### === Done: Quantify noise in signal - from phase stepping ===
    ### Requirement: phase-stepping measurement
    ### Calculation: from phase-stepping measurements
    ### Result: 'non-linearity in phase response is noise'-SNR

    ### === Quantify photobleaching ===
    ### Requirement: WFS experiment done, frames before and after WFS experiment
    ### Calculation: loss of intensity -> % photobleached
    ### Result: Value for amount of photobleaching
    ### Implementation complexity: 2
