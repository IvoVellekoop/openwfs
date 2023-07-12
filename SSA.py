from algorithm import Algorithm
import numpy as np
from base_device_properties import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options



class SSA(Algorithm):
    """
    Class definition for stepwise sequential algorithm.

    Has not been tested extensively and shows some unexpected behaviour
    """

    def __init__(self, p_steps =8 , initial_WF= 0,**kwargs):

        """
        Constructor.

        Parameters:
        -----------
        p_steps : int
            Number of phase steps used for measuring ideal incident wavefront.
        initial_WF : array-like
            Initial wavefront.
        """
        parse_options(self, kwargs)
        self._phase_steps = p_steps

        self.phase_set = np.round(np.arange(self._phase_steps) * 256 / self._phase_steps).astype(int)
    def init_wf(self):
        return np.zeros([self._n_slm_fields, self._n_slm_fields])
    def get_count(self):
        """
        Returns the total number of iterations.
        """
        return self.init_wf().size * self._phase_steps

    def get_wavefront(self, meas_count):
        """
        Calculates the incident wavefront (gray values) for the iteration number 'meas_count'.

        Parameters:
        -----------
        meas_count : int
            Iteration number.

        Returns:
        --------
        wavefront : array-like
            Incident wavefront (gray values).
        """
        wavefront = self.init_wf().copy()

        row_n = int(np.floor(meas_count / (wavefront.shape[1] * self._phase_steps)))  # SLM segment number
        collumn_n = int(np.floor(meas_count / self._phase_steps) % wavefront.shape[1])
        p = np.mod(meas_count, self._phase_steps)  # phase step number
        # we need to find which element this is in the array

        wavefront[row_n,collumn_n] = self.phase_set[p]
        return wavefront



    def post_process(self, feedback_set):
        """
        Optimize feedback signal by changing the phase of the incident wavefront.

        Parameters:
        -----------
        feedback_set : array-like
            Raw feedback signals.

        Returns:
        --------
        ideal_wavefronts : array-like
            Ideal phase for all SLM segments for all targets.
        t_set : array-like
            Calculated transmission coefficients for every segment for every target.
        feedback_set : array-like
            Raw feedback signals.
        """
        # Initialization
        M = feedback_set.shape[1]  # number of targets
        N = self.init_wf().size  # number of controlled SLM segments

        t_set = np.zeros((N, M), dtype=np.complex128)

        for m in range(M):
            feedback = feedback_set[:, m].reshape((self._phase_steps, N),order='F')

            # conceptually: this following line makes use of the fact that a full phase-stepping experiment
            # should result in a periodic signal. That signal has a certain phase (and a frequency of 1/p_steps),
            # and we try to find the phase such that the signal is maximal.

            product = feedback * np.exp(-1.0j * self.phase_set.reshape((-1, 1)) * ((2 * np.pi) / 256))
            t_set[:, m] = np.sum(product, axis=0) / self._phase_steps

        ideal_wavefronts = np.mod(np.round(np.angle(np.conj(t_set)) * 256 / (2 * np.pi)), 256)

        # Reshape size of ideal wavefronts to size input_wavefront
        ideal_wavefronts = ideal_wavefronts.reshape(self.init_wf().shape + (M,))


        return ideal_wavefronts, t_set, feedback
    phase_steps = int_property(min = 1, default=8)
    n_slm_fields = int_property(min=1, default=1)