# Built-in
from typing import Tuple

# External (3rd party)
import numpy as np
from numpy import ndarray as nd
from numpy.typing import ArrayLike

# External (ours)
from openwfs.devices import Camera
from openwfs.devices import SLM
from openwfs.simulation import SLM as MockSLM


class FringeAnalysisSLMCalibrator:
    """
    SLM calibrator that determines the field response using interference fringes.

    This calibrator requires an interferometer that produces interference fringes. A camera is used to observe the
    fringes. This camera must be conjugated to the SLM. The SLM is divided into two groups. One group is modulated using
    the gray value to be calibrated, the other group is used as a static reference. The SLM phase is then determined
    using the phase shift of the fringes.
    """

    def __init__(
        self,
        camera: Camera,
        slm: SLM | MockSLM,
        slm_mask: nd = None,
        modulated_slices: tuple[slice, slice] = None,
        reference_slices: tuple[slice, slice] = None,
        gray_values: ArrayLike = None,
        dc_skip: int = 4,
    ):
        """
        Args:
            camera: Camera that records the fringes.
            slm: SLM to be calibrated
            slm_mask: A 2D bool array of that defines the elements used by modulation group with True and elements used
                by reference group with False, should have shape ``(height, width)``.
            modulated_slices: Slice objects to crop the frame to the modulated fringes.
            reference_slices: Slice objects to crop the frame to the reference fringes.
            gray_values: Gray values to calibrate. Default: 0, 1, ..., 255.
            dc_skip: In the Fourier domain, a square region of this size (in Fourier-pixels) will be set to 0 to remove
                the DC peak, before determining the dominant frequency.
        """
        self.slm = slm
        self.camera = camera
        self.dc_skip = dc_skip

        if slm_mask is None:
            self.slm_mask = np.asarray(((True, True), (False, False)))
        else:
            self.slm_mask = slm_mask.astype(bool)

        if modulated_slices is None:
            self.modulated_slices = (slice(None), slice(0, self.camera.data_shape[0] // 3))
        else:
            self.modulated_slices = modulated_slices

        if reference_slices is None:
            self.reference_slices = (slice(None), slice(-self.camera.data_shape[0] // 3, None))
        else:
            self.reference_slices = reference_slices

        if gray_values is None:
            self.gray_values = np.arange(0, 255)
        else:
            self.gray_values = gray_values

    def execute(self) -> Tuple[nd, ArrayLike, nd]:
        frames = np.zeros((len(self.gray_values), *self.camera.data_shape))

        # Record a camera frame for every gray value
        for n, gv in enumerate(self.gray_values):
            self.slm.set_phases_8bit(self.slm_mask * gv)
            frames[n, ...] = self.camera.read()
            ### TODO: Can this be done like this? How does the camera behave? Test with real cam.
            # self.camera.trigger(out=frames[n, ...])

        self.camera.wait()
        return self.analyze(frames), self.gray_values, frames

    def analyze(self, frames):
        modulated_fringes = frames[:, self.modulated_slices[0], self.modulated_slices[1]]
        reference_fringes = frames[:, self.reference_slices[0], self.reference_slices[1]]

        modulated_fft = np.fft.fft2(modulated_fringes, axes=(-2, -1))
        reference_fft = np.fft.fft2(reference_fringes, axes=(-2, -1))

        modulated_dominant_freq = self.get_dominant_frequency(modulated_fft, self.dc_skip)
        reference_dominant_freq = self.get_dominant_frequency(reference_fft, self.dc_skip)

        relative_field = modulated_dominant_freq / reference_dominant_freq

        return relative_field

    @staticmethod
    def get_dominant_frequency(fft_data, dc_skip):
        fft_data[..., 0:dc_skip, 0:dc_skip] = 0             # Remove DC peak
        s = fft_data.shape
        reshaped_fft = fft_data.reshape(s[:-2] + (-1,))     # Flatten the last two axes (frequency dimensions) into one

        # Find the index of the maximum value along the last axis
        max_idx_flat = np.argmax(np.abs(reshaped_fft), axis=-1)
        max_idx_2d = np.unravel_index(max_idx_flat, (s[-2], s[-1]))

        # Compute indices to select the dominant frequency from the original array
        # This accounts for any leading dimensions
        indices = tuple(np.indices(s[:-2])) + max_idx_2d
        return fft_data[indices]
