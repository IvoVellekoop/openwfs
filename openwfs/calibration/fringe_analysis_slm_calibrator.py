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
        dc_skip: int = 5,
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
                the DC peak, before determining the dominant frequency. The image should contain significantly more
                fringes than this value.
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
            self.gray_values = np.arange(0, 256)
        else:
            self.gray_values = gray_values

    def execute(self) -> Tuple[nd, ArrayLike, nd]:
        dtype = self.camera.read().dtype                    # Read one frame to find dtype
        frames = np.zeros((len(self.gray_values), *self.camera.data_shape), dtype=dtype)

        # Record a camera frame for every gray value
        for n, gv in enumerate(self.gray_values):
            self.slm.set_phases_8bit(self.slm_mask * gv)
            frames[n, ...] = self.camera.read()

            ### TODO: Can reading frames be done like this? How does the camera behave? Test with real cam and slm.
            # self.camera.trigger(out=frames[n, ...])

        self.camera.wait()
        return self.analyze(frames), self.gray_values, frames

    def analyze(self, frames):
        modulated_fringes = frames[:, self.modulated_slices[0], self.modulated_slices[1]].astype(np.float32)
        reference_fringes = frames[:, self.reference_slices[0], self.reference_slices[1]].astype(np.float32)

        modulated_dominant_freq = self.get_dominant_frequency(modulated_fringes, self.dc_skip)
        reference_dominant_freq = self.get_dominant_frequency(reference_fringes, self.dc_skip)

        relative_field = modulated_dominant_freq / reference_dominant_freq

        return relative_field

    def get_dominant_frequency(self, img_data, dc_skip, std_factor=0.5):
        """
        Args:
            img_data: (Cropped) image data to perform Fourier fringe analysis on.
            dc_skip: Size in Fourier space pixels of the DC peak and surroundings to set to 0
                before finding dominant frequency.
            std_factor: Multiplier for the Gaussian window std to suppress edge effects. See gaussian_window
        """
        shape = img_data.shape[-2:]
        G = self.gaussian_window(shape, std_factor)

        fft_data = np.fft.fft2(G * img_data, axes=(-2, -1))

        fft_data[..., :dc_skip, :dc_skip] = 0  # Remove DC peak
        fft_data[..., :dc_skip, -dc_skip:] = 0  # Remove DC peak
        fft_data[..., -dc_skip:, -dc_skip:] = 0  # Remove DC peak
        fft_data[..., -dc_skip:, :dc_skip] = 0  # Remove DC peak

        # Find the index of the maximum value along the last axis
        max_idx_flat = np.argmax(np.abs(fft_data[0, ...]))
        max_idx_2d = np.unravel_index(max_idx_flat, shape)

        # Compute indices to select the dominant frequency from the original array
        # This accounts for any leading dimensions
        return fft_data[:, *max_idx_2d]

    @staticmethod
    def gaussian_window(shape, std_factor):
        """
        Generates a 2D Gaussian window of requested shape.

        Args:
          shape: Shape of the Gaussian window
          std_factor: Standard deviation multiplier. (1 -> std = window size)

        Returns:
          2D numpy array of shape (M, N), the Gaussian window
        """
        M, N = shape

        # Standard deviations
        std_x = (N/2) * std_factor
        std_y = (M/2) * std_factor

        # Coordinate arrays centered at zero
        x = np.arange(N).reshape(1, -1) - (N-1)/2
        y = np.arange(M).reshape(-1, 1) - (M-1)/2

        # Return Gaussian window
        return np.exp(-((x**2) / (2 * std_x**2) + (y**2) / (2 * std_y**2)))
