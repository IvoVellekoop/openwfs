import numpy as np
import matplotlib.pyplot as plt
import cv2

from base_device_properties import *


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class SimulatedWFS:
    """A Simulated 2D wavefront shaping experiment. Has a settable ideal wavefront, and can calculate a feedback
    image for a certain SLM pattern.

    It is both an SLM and a camera, so it can be loaded as both in the WFS algorithms for testing.

    Todo: the axis of the SLM & image plane are bogus, they should represent real values
    """

    def __init__(self, shape=(500, 500), active_plotting=False, **kwargs):

        self._active_plotting = active_plotting
        self.resized = True
        self.phases = np.zeros(shape)
        parse_options(self, kwargs)

        if self.active_plotting:
            plt.figure(1)
            plt.get_current_fig_manager().window.setGeometry(50, 100, 640, 545)
            self.slm_plot = plt.imshow(self.phases)
            plt.colorbar()
            plt.clim(0, 256)
            plt.figure(2)
            plt.get_current_fig_manager().window.setGeometry(750, 100, 640, 545)
            self._image_plot = plt.imshow(self.image)
            self._image = self.get_image()
            self._image_plot = plt.imshow(self.get_image())
            plt.xlim([245, 255])
            plt.ylim([245, 255])
            plt.ion()  # Turn on interactive mode

        self.E_input_slm = make_gaussian(shape[0], fwhm=self.beam_profile_fwhm)
        self.ideal_wf = np.zeros(shape)

        self.max_intensity = 1
        self.t_idle = 0
        self.t_settle = 0

    def trigger(self):
        pass

    def reserve(self, time_ms):
        pass

    def wait(self):
        "This is where the image gets created and put in the pre-existing buffer"
        pattern = cv2.resize(self.phases, dsize=np.shape(self.ideal_wf), interpolation=cv2.INTER_NEAREST)

        field_slm = self.E_input_slm * np.exp(1j * (pattern - (self.ideal_wf / 256 * 2 * np.pi)))
        field_slm_f = np.fft.fft2(field_slm)

        image_plane = np.array(abs(np.fft.fftshift(field_slm_f) ** 2))

        # the max intensity must be the highest found intensity, and at least 1.
        self.max_intensity = np.max([np.max(image_plane), self.max_intensity, 1])
        self.image[:, :] = np.array((image_plane / self.max_intensity) * (2 ** 16 - 1), dtype=np.uint16)

        pass

    def read(self):
        self.wait()
        return self.image

    def update(self, wait=1.0):
        pass

    def wait_stable(self):
        pass

    @property
    def measurement_time(self):
        return self.exposure_ms

    @property
    def data_shape(self):
        return self.height, self.width

    def set_data(self, pattern):
        self.phases = pattern / 256 * 2 * np.pi  # convert from legacy format to new format
        if self._active_plotting:
            self.slm_plot.set_data(self.phases)
            plt.pause(0.001)
            plt.draw()

    def set_activepatch(self, id):
        pass

    def set_rect(self, rect):
        pass

    def destroy(self):
        pass

    def set_ideal_wf(self, ideal_wf):
        ideal_wf = cv2.resize(ideal_wf, dsize=np.shape(self.ideal_wf), interpolation=cv2.INTER_NEAREST)
        self.ideal_wf = ideal_wf

    # def get_image(self):
    #     return np.zeros((self._width, self._height), dtype=np.uint16)

    def get_image(self):
        "Where the buffer is made and passed. Not the place where the imaging should occur"
        if self.resized:
            self._image = np.zeros((self.width, self.height), dtype=np.uint16)
            self.resized = False

        if self._active_plotting:
            self._image_plot.set_data(self._image)
            plt.pause(0.001)
            plt.draw()

        return self._image

    # required for camera implementation
    exposure_ms = float_property(min=0.0, default=100)
    top = int_property(min=-1000, max=5000, default=0)
    left = int_property(min=-1000, max=5000, default=0)
    width = int_property(min=500, max=500, default=500)
    height = int_property(min=500, max=500, default=500)
    Binning = int_property(min=1, default=1)
    image = property(fget=get_image)
    wavelength_nm = float_property(min=400, default=804, max=1600)
    beam_profile_fwhm = float_property(min=1, default=100)
    # useful
    active_plotting = bool_property(default=0)


def generate_double_pattern(shape, phases_half1, phases_half2, phase_offset):
    width, height = shape
    half_width = width // 2

    # Generate halves the image with phase offset
    if phases_half1 == 0:
        half1 = [[0 for _ in range(half_width)] for _ in range(height)]
    else:
        half1 = [[int(((i + phase_offset) / (width / phases_half1)) * 256) % 256 for _ in range(half_width)] for i in
                 range(height)]

    if phases_half2 == 0:
        half2 = [[0 for _ in range(half_width)] for _ in range(height)]
    else:
        half2 = [[int((i / (width / phases_half2)) * 256) % 256 for _ in range(half_width)] for i in range(height)]

    # Combine both halves
    image_array = [row1 + row2 for row1, row2 in zip(half1, half2)]

    # Resize the image to the specified shape
    image_array = [row[:width] for row in image_array[:height]]

    return np.array(image_array)


# experiment
if __name__ == "__main__":
    shape = [500, 500]
    exp = SimulatedWFS(shape=shape, active_plotting=False)

    exp.set_data(generate_double_pattern([500, 500], 20, 0, 0))

    # plotting results
    plt.figure()
    plt.imshow(exp.phases)
    plt.figure()
    exp.wait()
    plt.imshow(exp.image)
    print(exp.get_image()[250, 250])
    plt.show()
