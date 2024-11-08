# External (3rd party)
import numpy as np

# External (ours)
from openwfs import Detector, PhaseSLM


class InlineSLMCalibrator:
    """
    Inline SLM Calibrator

    Calibrate an SLM phase response based on feedback measurements.
    """

    def __init__(self, feedback: Detector, slm: PhaseSLM, num_of_phase_steps=256):
        self.feedback = feedback
        self.slm = slm
        self.num_of_phase_steps = num_of_phase_steps
        self.phase_response = None

    def execute(self):
        # TODO: must be updated to work
        Nx = 4
        Ny = 4
        static_phase = 0

        # Create checkerboard phase mask
        y_check = np.arange(Ny).reshape((Ny, 1))
        x_check = np.arange(Nx).reshape((1, Nx))
        checkerboard = np.mod(x_check + y_check, 2)
        static_phase_pattern = static_phase * (1 - checkerboard)

        # Random SLM pattern to destroy focus
        self.slm.set_phases(2 * np.pi * np.random.rand(300, 300))
        self.slm.update()

        # Read dark frame
        dark_frame = self.feedback.read()
        dark_var = dark_frame.var()

        count = 0
        data_shape = self.feedback.data_shape
        img_stack = np.zeros((data_shape[0], data_shape[1], self.num_of_phase_steps))
        phase_range = np.linspace(0, 2 * np.pi, self.num_of_phase_steps)
        for phase in phase_range:
            phase_pattern = static_phase_pattern + phase * checkerboard
            self.slm.set_phases(phase_pattern)
            self.slm.update()

            img_stack[:, :, count] = self.feedback.read().copy()
            count += 1

        self.slm.set_phases(2 * np.pi * np.random.rand(300, 300))
        self.slm.update()

        # Plot frames
        # fig = plt.figure()
        # for n in range(num_of_phase_steps):
        #     fig.clear()
        #     plt.imshow(img_stack[:, :, n].squeeze())
        #     plt.title(f'{n}')
        #     plt.draw()
        #     plt.pause(1e-3)

        # STD of frame, corrected for background noise
        stack_std_corrected = np.sqrt((img_stack.var(axis=(0, 1)) - dark_var).clip(min=0))
        block_size_pix = int(slm.shape[0] / Ny)
        return self.phase_response
