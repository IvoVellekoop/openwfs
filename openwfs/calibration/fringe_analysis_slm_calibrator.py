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
        camera_mask_modulate: nd = None,
        camera_mask_reference: nd = None,
        gray_values: ArrayLike = None,
    ):
        """
        Args:
            camera: Camera that records the fringes.
            slm: SLM to be calibrated
            slm_mask: A 2D bool array of that defines the elements used by modulation group with True and elements used
                by reference group with False, should have shape ``(height, width)``.
            line_pick_axis: Axis from which to pick a horizontal or vertical line. The other axis will be used to
                perform Fourier fringe analysis on. E.g. vertical fringes -> choose the vertical axis, as the FFT must
                be taken along the horizontal axis.
            line_index_modulate: Frame pixel index to pick a horizontal or vertical line to analyze, for modulation.
                Default: 0.25 * frame width or height.
            line_index_reference: Frame pixel index to pick a horizontal or vertical line to analyze, for reference.
                Default: 0.75 * frame width or height.
            gray_values: Gray values to calibrate. Default: 0, 1, ..., 255.
        """
        self.slm = slm
        self.camera = camera

        if slm_mask is None:
            self.slm_mask = np.asarray(((True, True), (False, False)))
        else:
            self.slm_mask = slm_mask.astype(bool)

        if gray_values is None:
            self.gray_values = np.arange(0, 255)
        else:
            self.gray_values = gray_values

    def execute(self):
        frames = np.zeros((len(self.gray_values), *self.camera.data_shape))

        # Record a camera frame for every gray value
        for n, gv in enumerate(self.gray_values):
            self.slm.set_phases_8bit(self.slm_mask * gv)
            frames[:, :, n] = self.camera.trigger(out=frames[n, ...])

        self.camera.wait()
        return self.analyze(frames), frames

    def analyze(self, frames):
        modulated_fringes = np.take(frames, self.line_index_modulate, axis=self.line_pick_axis)
        reference_fringes = np.take(frames, self.line_index_reference, axis=self.line_pick_axis)

        np.fft.fft2(modulated_fringes, axes=(-2, -1))
        np.fft.fft2(reference_fringes, axes=(-2, -1))

        return fields
