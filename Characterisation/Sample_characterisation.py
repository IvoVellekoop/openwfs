import numpy as np
import sys

from base_device_properties import float_property, int_property, string_property, object_property, base_property, \
    bool_property, parse_options


class FrequencyCharacterisation():

    def __init__(self, **kwargs):
        """
        constructor
        """
        parse_options(self, kwargs)

    def frequency_scan(self):

        self.scan_x = np.arange(self.frequency_min, self.frequency_max + 1, 1)
        self.scan_y = np.array([0])
        self.wfs.algorithm.set_kspace(self.scan_x, self.scan_y)
        fourier.phase_steps = 8

        self.wfs.execute = 1
        self.frequency_response = self._calculate_frequency_response(self.wfs.feedback_set, self.scan_x, self.scan_y)

    def _calculate_frequency_response(self, feedback, k_xset, k_yset, split_lr=False):

        """ This function calculates the frequency response of a wavefront shaping experiment.

        It does this by reading the feedback signal from the experiment and fourier-transforming the signal for each phase set.
        """

        feedback_shape = np.shape(feedback)
        feedbackleft = feedback[:, :, 0].reshape(feedback_shape[1], feedback_shape[0], order='C')[
                       :feedback_shape[1] // 2, :]
        feedbackright = feedback[:, :, 0].reshape(feedback_shape[1], feedback_shape[0], order='C')[
                        feedback_shape[1] // 2:, :]
        signalleft = np.abs(np.fft.fft(feedbackleft, axis=1))
        signalleft = np.max(signalleft, axis=1)

        signalright = np.abs(np.fft.fft(feedbackright, axis=1))
        signalright = np.max(signalright, axis=1)

        signal = signalleft + signalright

        signal_strength_grid = signal.reshape((len(k_xset), len(k_yset)))

        if split_lr:
            return signalleft.reshape((len(k_xset), len(k_yset))), signalright.reshape((len(k_xset), len(k_yset)))
        else:
            return signal_strength_grid.T

    def find_top_frequencies(self):
        frequencies = self._find_top_n(self.frequency_response, self.scan_x, self.n_frequencies)
        frequencies = np.unique(np.append(frequencies, [-1, 0, 1]))

        if len(frequencies) != self.n_frequencies:
            n_over = len(frequencies) - self.n_frequencies
            frequencies = self._find_top_n(self.frequency_response, self.scan_x, self.n_frequencies - n_over)
            frequencies = np.unique(np.append(frequencies, [-1, 0, 1]))

        self.kx_set = np.sort(frequencies)
        self.ky_set = np.sort(frequencies)
        return self.kx_set, self.ky_set

    def _find_top_n(self, signal_strength_grid, kset, n):
        # Flatten the signal strength grid
        flattened_grid = signal_strength_grid.flatten()

        # Find the indices of the top n highest values
        indices = np.argpartition(flattened_grid, -n)[-n:]

        # Get the associated k_xset values
        associated_values = kset[indices]

        return associated_values

    def set_kspace(self):
        self.wfs.algorithm.set_kspace(self.kx_set, self.ky_set)

    phase_steps = int_property(min=1, default=8)
    frequency_min = int_property(default=-20)
    frequency_max = int_property(default=20)
    n_frequencies = int_property(default=10)
    wfs = object_property()