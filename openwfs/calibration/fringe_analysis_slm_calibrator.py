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
        """
        self.slm = slm
        self.camera = camera

        if slm_mask is None:
            self.slm_mask = np.asarray(((True, True), (False, False)))
        else:
            self.slm_mask = slm_mask.astype(bool)

        if modulated_slices is None:
            self.modulated_slices = (slice(0, self.camera.data_shape[0] // 4), slice(None))
        else:
            self.modulated_slices = modulated_slices

        if reference_slices is None:
            self.reference_slices = (slice(-self.camera.data_shape[0] // 4, None), slice(None))
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
            self.camera.trigger(out=frames[n, ...])

        self.camera.wait()
        return self.analyze(frames), self.gray_values, frames

    def analyze(self, frames):
        modulated_fringes = frames[:, self.modulated_slices[0], self.modulated_slices[1]]
        reference_fringes = frames[:, self.reference_slices[0], self.reference_slices[1]]

        modulated_fft = np.fft.fft2(modulated_fringes, axes=(-2, -1))
        reference_fft = np.fft.fft2(reference_fringes, axes=(-2, -1))

        modulated_dominant_freq = self.get_dominant_frequency(modulated_fft)
        reference_dominant_freq = self.get_dominant_frequency(reference_fft)

        relative_field = modulated_dominant_freq / reference_dominant_freq

        return relative_field

    @staticmethod
    def get_dominant_frequency(fft_data, dc_skip=4):
        # Calculate the Nyquist frequencies
        nyquist_y = fft_data.shape[-2] // 2
        nyquist_x = fft_data.shape[-1] // 2

        # Input data was real -> we're only interested in one half of the data
        # Also remove DC peak
        cropped_fft = fft_data[..., dc_skip + 1 : nyquist_y, dc_skip + 1 : nyquist_x]

        # Get the shape of the cropped FFT data
        cropped_shape = cropped_fft.shape

        # Flatten the last two axes (frequency dimensions) into one
        reshaped_fft = cropped_fft.reshape(cropped_shape[:-2] + (-1,))

        # Find the index of the maximum value along the last axis
        max_idx_flat = np.argmax(np.abs(reshaped_fft), axis=-1)

        # Convert the flat indices back to 2D indices
        ny, nx = cropped_shape[-2], cropped_shape[-1]
        max_idx_2d = np.unravel_index(max_idx_flat, (ny, nx))

        # Prepare indices to select the dominant frequency from the original array
        # This accounts for any leading dimensions (e.g., frames)
        indices = tuple(np.indices(cropped_shape[:-2])) + max_idx_2d

        # Retrieve the dominant frequency values
        dominant_freq = cropped_fft[indices]

        return dominant_freq
