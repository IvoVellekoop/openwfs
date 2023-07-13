from algorithm import Algorithm
import numpy as np
from base_device_properties import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options



class FourierDualRef(Algorithm):
    """
    The 2 step optimization algorithm for the wavefront in Fourier-space.
    Half of the SLM are modulated at the same time, the other half is kept as a reference

    Has methods build_kspace and set_kspace. Build kspace makes an evenly spaced kspace from the parameters of the class
    and set_kspace can be any kspace, for more advanced algorithms.

    Will only work for rectangular geometries
    """

    def __init__(self,**kwargs):
        """
        constructor
        """
        self.circular = 0
        parse_options(self,kwargs)
        self.build_kspace(1)


    def build_kspace(self,value=1):
        kx_set = np.arange(self.kx_angles_min, self.kx_angles_max + self.kx_angles_stepsize, self.kx_angles_stepsize)
        ky_set = np.arange(self.ky_angles_min, self.ky_angles_max + self.ky_angles_stepsize, self.ky_angles_stepsize)

        self.set_kspace(kx_set,ky_set)
        return value

    phase_steps = int_property(min = 1, default=8)
    kx_angles_min = int_property(default=-3, on_update=build_kspace)
    kx_angles_max = int_property(default=3, on_update=build_kspace)
    kx_angles_stepsize = int_property(default=1, on_update=build_kspace)
    ky_angles_min = int_property(default=-3, on_update=build_kspace)
    ky_angles_max = int_property(default=3, on_update=build_kspace)
    ky_angles_stepsize = int_property(default=1, on_update=build_kspace)
    overlap_coeficient = float_property(min=0, max=1, default=0.1)




    def set_kspace(self,kx_set,ky_set):
        Nx = np.round(np.array(self.init_wf().shape) / 2) * 2

        if Nx[0] == 1 or Nx[1] == 1:
            raise Exception('Fourier algorithm can only be used with a rectangular SLM geometry')





        if self.circular:
            self.kset_circular = 1  # k values in the corners will be excluded, and k space will be in a circular shape
        else:
            self.kset_circular = 0  # k space will be in a rectangular shape

        # set normalized wavefront coordinates
        self.x = np.arange(0, Nx[1]) / Nx[1]
        self.y = np.arange(0, Nx[0] / 2 + self.overlap_coeficient * Nx[0] / 2) / Nx[0]

        # set kx and ky components
        self.Nk_full = len(kx_set) * len(ky_set)
        kx_set_total = np.repeat(np.array(kx_set)[np.newaxis, :], len(ky_set), axis=0).flatten()
        ky_set_total = np.repeat(np.array(ky_set)[:, np.newaxis], len(kx_set), axis=1).flatten()
        k_max = max(kx_set)

        if self.kset_circular:
            # find the effective area in k space
            k_index = np.where(kx_set_total ** 2 + ky_set_total ** 2 <= k_max ** 2)[0]
            self.kx_set = kx_set_total[k_index]
            self.ky_set = ky_set_total[k_index]

        else:
            # using the whole k space
            k_index = np.arange(0, self.Nk_full)
            self.kx_set = kx_set_total
            self.ky_set = ky_set_total

        self.Nk = len(k_index)
        self.k_index = k_index
        # set other algorithm properties
        initial_WF = self.init_wf()
        self.center = initial_WF.shape[0]//2
        self.overlap_start = int(self.center*(1 - self.overlap_coeficient))
        self.overlap_end = initial_WF.shape[0] - self.overlap_start - 1

        self.phase_set = np.round(np.arange(0, self.phase_steps) * 256 / self.phase_steps).astype(int)


    def init_wf(self):
        return np.zeros([1056, 1056])
    def get_count(self):
        # returns the total number of iterations
        return 2*self.Nk*self.phase_steps

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
        k = np.mod(np.floor(meas_count / self.phase_steps).astype(int), self.Nk)
        p = np.mod(meas_count, self.phase_steps)

        if meas_count <= self.get_count() / 2:
            active = np.arange(self.y.size)
        else:
            active = slice(self.overlap_start, None)

        wavefront[:, active] = self.phase_set[p] + 256 * (
                    self.x[:,np.newaxis] * self.kx_set[k] + self.y[np.newaxis,:] * self.ky_set[k])

        return wavefront

    def post_process(self, feedback_set_in):
        """
        Optimize feedback signal by changing the phase of the incident wavefront.

        Parameters:
        -----------
        feedback_set_in : array-like
            Raw feedback signals.

        Returns:
        --------
        ideal_wavefront : array-like
            Ideal phase for all SLM segments for all targets.
        Tset : array-like
            Calculated transmission coefficients for every segment for every target.
        feedback_set_out : array-like
            Raw feedback signals.
        """

        dim_feedback_set = feedback_set_in.shape
        dim_for_ROIs = 1  # the first dimension is assumed to be the ROI index
        possible_partitions = np.array([i for i in range(1, dim_feedback_set[dim_for_ROIs]+1) if dim_feedback_set[dim_for_ROIs] % i == 0])
        divisor_count = 3
        if len(possible_partitions) > divisor_count:
            number_of_partitions = possible_partitions[divisor_count - 1]
        else:
            print("Could not subdivide the feedback set. Using the whole set at once.")
            number_of_partitions = 1

        roi_size = dim_feedback_set[dim_for_ROIs] // number_of_partitions

        ideal_wavefront = np.zeros((self.Nk_full, self.init_wf().shape[0], self.init_wf().shape[1]))
        Tset = np.zeros((self.Nk_full, self.init_wf().shape[0], self.init_wf().shape[1]))

        for part_idx in range(number_of_partitions):
            roi_start = part_idx * roi_size
            roi_end = (part_idx + 1) * roi_size if part_idx != number_of_partitions - 1 else dim_feedback_set[dim_for_ROIs]
            roi_indices = np.arange(roi_start, roi_end)

            feedback_set_part = feedback_set_in[:, roi_indices]

            for roi_idx in roi_indices:
                feedback_set_out, ideal_wavefront_roi, Tset_roi = self._process_roi_feedback(feedback_set_part[:, roi_idx - roi_start])

                ideal_wavefront = ideal_wavefront_roi
                Tset = Tset_roi

        return ideal_wavefront, Tset, feedback_set_out

    def _process_roi_feedback(self, feedback_set_roi):
        """
        This function optimizes the feedback signal by changing the phase of the incident wavefront and calculates the
        transmission matrix.

        Parameters:
            feedback_set_roi (np.ndarray): Feedback signal in the region of interest (ROI).

        Returns:
            feedback_set_roi (np.ndarray): Feedback signal in the region of interest (ROI).
            ideal_wavefront_roi (np.ndarray): Ideal wavefronts in the ROI.
            Tset_roi (dict): Transmission matrices.
        """

        # Calculate transmission matrix
        t1, t1_fourier = self._compute_correction(feedback_set_roi[:feedback_set_roi.shape[0] // 2])
        t2, t2_fourier = self._compute_correction(feedback_set_roi[feedback_set_roi.shape[0] // 2:])

        ind_overlap = slice(self.overlap_start, self.overlap_end)

        # Optimize feedback signal by changing the phase of the incident wavefront
        if feedback_set_roi.ndim == 1: # this is doubly implemented
            M = 1
        else:
            M = feedback_set_roi.shape[1]

        for m in range(M):
            overlap_len = len(range(ind_overlap.start, ind_overlap.stop))
            t11 = t1[:, -overlap_len:, m]
            t22 = t2[:, :overlap_len, m]
            c = np.vdot(t22, t11)
            factor = c / abs(c) * np.linalg.norm(t11) / np.linalg.norm(t22)
            t2[:, :, m] = t2[:, :, m] * factor

        t_overlap = (t2[:, :overlap_len, :] + t1[:, ind_overlap, :]) / 2
        t_set = np.concatenate(
            [t1[:, :-overlap_len, :], t_overlap, t2[:,self.overlap_end - self.overlap_start + 1:, :]], axis=1)

        ideal_wavefront_roi = np.mod(np.round(np.angle(np.conj(t_set)) * 256 / (2 * np.pi)), 256)

        # Reshape size of ideal wavefronts to size input_wavefront
        feedback_set_roi = feedback_set_roi.reshape((self.phase_steps, 2 * self.Nk, M))
        Tset_roi = {'t_set': t_set}
        if self.kset_circular:
            Tset_roi['t1f'] = np.zeros((1, self.Nk_full, M))
            Tset_roi['t2f'] = np.zeros((1, self.Nk_full, M))
            for m in range(M):
                Tset_roi['t1f'][0, self.k_index, m] = t1_fourier[:, :, m]
                Tset_roi['t2f'][0, self.k_index, m] = t2_fourier[:, :, m]
            Tset_roi['N_effective'] = 2 * len(self.k_index)
        else:
            Tset_roi['t1f'] = t1_fourier
            Tset_roi['t2f'] = t2_fourier

        return feedback_set_roi, ideal_wavefront_roi, Tset_roi

    def _compute_correction(self, feedback_subset):
        if feedback_subset.ndim == 1:
            M = 1
        else:
            M = feedback_subset.shape[1]
        t = np.zeros((1, self.Nk, M), dtype=np.complex128)
        feedback = feedback_subset.reshape(
            (self.phase_steps, self.Nk, M),order ='F')  # order = f due to row-column major issues during matlab translation

        # number of steps by number of k-set by number of targets
        Ecorr = np.zeros((len(self.x), len(self.y), M), dtype=np.complex128)

        for m in range(M):
            # extract transmission matrix elements from feedback signal
            t[0, :, m] = np.sum(feedback[:, :, m] * np.exp(-1.0j * self.phase_set.reshape((-1,1)) * (2 * np.pi) / 256),
                                axis=0) / self.phase_steps


            # compute correction field by superimposing all measured wavefronts
            gradient = (self.x.reshape(-1, 1, 1) * self.kx_set.reshape(1, 1, -1)
                        + self.y.reshape(1, -1, 1) * self.ky_set.reshape(1, 1, -1))
            Ecorr[:, :, m] = np.sum(
                np.reshape(t[:, :, m], (1, 1, self.Nk)) * np.exp(-2.0j * np.pi * gradient),
                axis=2
            )
        return Ecorr, t