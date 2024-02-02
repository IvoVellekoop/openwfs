import time

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def troubleshoot(algorithm, frame_source, light_on, light_off):
    """
    Run a series of basic checks to find common sources of error in a WFS experiment.
    Quantifies several types of fidelity reduction.
    """

    ### An outline of the to be written troubleshooting code.
    ### Each block indicates a step; i.e. a particular action or calculation, and may depend on
    ### previous steps (e.g. darkframe measurement). This indication helps with selecting the
    ### required order of steps.
    ### All steps where images are taken, must take images of the same part of the sample and
    ### some features in the image must be visible.
    ### TODO: some flags to indiciate which troubleshooting functions should run (especially ability to turn off
    ### long measurements)

    # Capture frames before WFS
    # Dark frame
    # Before frame

    # Frame CNR

    if True:  # TODO: Add flag
        # WFS experiment
        recompute_wf_flag = self.recompute_wavefront
        self.recompute_wavefront = True
        self.wavefront = WFSController.State.SHAPED_WAVEFRONT
        self.recompute_wavefront = recompute_wf_flag

        # Capture frames after WFS
        after_frame_flatwf = self.read_after_frame_flatwf()  # After-frame flat wavefront
        after_frame_shapedwf = self.read_after_frame_shapedwf()  # After-frame shaped wavefront

        self._contrast_enhancement = contrast_enhancement(after_frame_shapedwf, after_frame_flatwf, dark_frame)

    # Test setup stability
    pixel_shifts, correlations = self.test_setup_stability(1, 5)

    ### Debugging
    import matplotlib.pyplot as plt
    plt.plot(pixel_shifts[:, 0], label='x')
    plt.plot(pixel_shifts[:, 1], label='y')
    plt.plot(correlations, label='Corr. with first')
    plt.legend()
    plt.show()

    ### === Test setup stability ===
    ### Requirement: find-image-pixel-shift function
    ### Preconfig: Flat wavefront
    ### Measurement: Snap multiple images over a long period of time
    ### Calculation: Cross-correlation between images
    ### Result: xdrift, ydrift, intensity drift, warning if >threshold
    ### Note 1: measurement should take a long time, could result in significant photobleaching
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


class WFSTroubleshootResult:
    """
    Data structure for holding wavefront shaping results, statistics, and additional troubleshooting information.
    """
    def __init__(self):
        self.fidelity_non_modulated = None
        self.fidelity_phase_calibration = None
        self.fidelity_phase_jitter = None


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


def measure_setup_stability(self, wait_time_s, num_of_frames):
    """Test the setup stability by repeatedly reading frames."""
    first_frame = self.read_frame()
    pixel_shifts = np.zeros(shape=(num_of_frames, 2))
    correlations = np.zeros(shape=(num_of_frames,))

    for n in range(num_of_frames):
        # self.set_slm_random()
        time.sleep(wait_time_s)
        # self._wavefront = self.State.FLAT_WAVEFRONT

        new_frame = self.read_frame()
        pixel_shifts[n, :] = find_pixel_shift(first_frame, new_frame)
        correlations[n] = frame_correlation(first_frame, new_frame)

    return pixel_shifts, correlations
