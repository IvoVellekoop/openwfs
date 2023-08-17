import math
import numpy as np
from typing import Any, Annotated
import matplotlib.pyplot as plt
import os
import pickle


class FourierDualRef:
    """
    Base class definition for fourier algorithm. As described by Mastiani et al. [1]

    Can run natively, provided you input a k_set.
    the k-set currently should have the shape of [2,n] with [0,:] = k_x and [1,:] = k_y.

    [1]: Bahareh Mastiani, Gerwin Osnabrugge, and Ivo M. Vellekoop,
    "Wavefront shaping for forward scattering," Opt. Express 30, 37436-37445 (2022)
    """

    def __init__(self, k_set=None, phase_steps=4, overlap=0.1, controller=None):
        """
        k_set: the [kx,ky] matrix 2 x n. #Maybe this should be split into an kx_set and ky_set such that the left & right
        SLM sides can be split
        """
        self._controller = controller
        self._phase_steps = phase_steps
        self._overlap = overlap

        if k_set is not None:
            self.k_x = k_set[0, :]
            self.k_y = k_set[1, :]

    def execute(self):
        self.experiment()

        t_fourier = self.controller.compute_transmission(self.phase_steps)

        t_slm = self.compute_t(t_fourier)

        return t_slm

    def experiment(self):
        self.controller.reserve((len(self.k_x) * 2, self.phase_steps))

        phases = np.arange(self.phase_steps) / self.phase_steps * 2 * np.pi
        for side in range(2):
            for n_angle in range(len(self.k_x)):
                for phase in phases:
                    self.controller.slm.phases = self.get_phase_pattern(self.k_x[n_angle], self.k_y[n_angle], phase,
                                                                        side)
                    self.controller.measure()

    def get_phase_pattern(self, kx, ky, p, side):
        height = self.controller.slm.width
        width = self.controller.slm.height
        overlap = int(self._overlap * width // 2)
        start = 0 if side == 0 else 0.5 - self._overlap / 2
        end = 0.5 + self._overlap / 2 if side == 0 else 1

        x = np.arange(start, end, 1 / ((height - overlap) + overlap))[np.newaxis, :]
        y = np.arange(0, 1, 1 / width)[:, np.newaxis]

        final_pattern = np.zeros((width, height))

        if side == 0:
            final_pattern[:, :x.shape[1]] = (2 * np.pi * kx) * x + ((2 * np.pi * ky) * y + p)
        else:
            final_pattern[:, -x.shape[1]:] = (2 * np.pi * kx) * x + ((2 * np.pi * ky) * y + p)

        return final_pattern

    def compute_t(self, t_fourier):
        """
        Computes the transmission matrix of the measurements,
        for the left and right side of the SLM separately and then combines them with the overlap.
        """
        # bepaal ruis: bahareh. Find peak & dc ofset
        t1 = np.zeros((self.controller.slm.height, self.controller.slm.width), dtype='complex128')
        t2 = np.zeros((self.controller.slm.height, self.controller.slm.width), dtype='complex128')

        n_experiments = int(np.floor(len(t_fourier) / 2))
        for n, t in enumerate(t_fourier):
            if n < n_experiments:
                phi = self.get_phase_pattern(self.k_x[n], self.k_y[n], 0, 0)
                t1 += np.exp(1j * phi) * np.conj(t)
            else:
                phi = self.get_phase_pattern(self.k_x[n - n_experiments], self.k_y[n - n_experiments], 0, 1)
                t2 += np.exp(1j * phi) * np.conj(t)

        overlap_len = int(self._overlap * self.controller.slm.width)
        overlap_begin = self.controller.slm.width // 2 - int(overlap_len / 2)
        overlap_end = self.controller.slm.width // 2 + int(overlap_len / 2)

        if self._overlap != 0:
            c = np.vdot(t2[:, overlap_begin:overlap_end], t1[:, overlap_begin:overlap_end])
            factor = c / abs(c) * np.linalg.norm(t1[:, overlap_begin:overlap_end]) / np.linalg.norm(t2[:, overlap_begin:overlap_end])
            t2 = t2 * factor

            overlap = (t1[:, overlap_begin:overlap_end] + t2[:, overlap_begin:overlap_end]) / 2
            t_full = np.concatenate([t1[:, 0:overlap_begin], overlap, t2[:, overlap_end:]], axis=1)
        else:
            t_full = np.concatenate([t1[:, 0:overlap_begin], t2[:, overlap_end:]], axis=1)


        return t_full

    @property
    def phase_steps(self) -> int:
        return self._phase_steps

    @phase_steps.setter
    def phase_steps(self, value):
        self._phase_steps = value

    @property
    def controller(self) -> Any:
        return self._controller

    @controller.setter
    def controller(self, value):
        self._controller = value


class BasicFDR(FourierDualRef):
    """
    The most simple implementation of the FourierDualRef algorithm. It constructs a symmetric k-space for the algorithm.
    The k-space initializer is set to None, because for custom k-spaces you should use FourierDualRef directly.
    """
    def __init__(self, phase_steps=4, k_angles_min=-3, k_angles_max=3, overlap=0.1, controller=None):
        super().__init__(None, phase_steps, overlap, controller)
        self._k_angles_min = k_angles_min
        self._k_angles_max = k_angles_max

        self.build_kspace()

    def build_kspace(self):
        kx_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        ky_angles = np.arange(self._k_angles_min, self._k_angles_max + 1, 1)
        # Make  the carthesian product of kx_angles and ky_angles to make a square kspace

        self.k_x = np.repeat(np.array(kx_angles)[np.newaxis, :], len(ky_angles), axis=0).flatten()
        self.k_y = np.repeat(np.array(kx_angles)[:, np.newaxis], len(ky_angles), axis=1).flatten()

    @property
    def k_angles_min(self) -> int:
        return self._k_angles_min

    @k_angles_min.setter
    def k_angles_min(self, value):
        self._k_angles_min = value

    @property
    def k_angles_max(self) -> int:
        return self._k_angles_max

    @k_angles_max.setter
    def k_angles_max(self, value):
        self._k_angles_max = value


def get_neighbors(n, m):
    """Get the neighbors of a point in a 2D grid."""
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),   (1, 0),  (1, 1)
    ]

    neighbors = [(n + dx, m + dy) for dx, dy in directions]
    return np.array(neighbors)


class CharacterisingFDR(FourierDualRef):
    """
     implementation of the FourierDualRef algorithm.
    """
    def __init__(self, phase_steps=4, overlap=0.1, max_modes=20, high_modes=0, high_phase_steps=16, controller=None):
        super().__init__(None, phase_steps, overlap, controller)
        self.max_modes = max_modes
        self.high_modes = high_modes
        self.high_phase_steps = high_phase_steps
        self.t_left = None
        self.t_right = None
        self.k_left = None
        self.k_right = None
        self.t_slm = None
        self.intermediate_enhancements = []
        self.added_modes = [['Uncorrected enhancement']]

    def execute(self):
        kx_total = []
        ky_total = []

        # measure the flat_wf value
        self.record_intermediate_enhancement([0], [0], [0], 0)

        for side in range(2):
            n = 0
            # we begin in K[0,0]
            self.k_x = np.zeros((1, 1), dtype=int)
            self.k_y = np.zeros((1, 1), dtype=int)
            t_fourier = np.array([])
            kx_total = np.array([])
            ky_total = np.array([])
            centers = np.array([[], []], dtype=int)

            while n < self.max_modes:

                self.single_side_experiment(side)
                t_fourier = np.append(t_fourier,self.controller.compute_transmission(self.phase_steps))
                kx_total = np.append(kx_total, self.k_x)
                ky_total = np.append(ky_total, self.k_y)

                found_next_center = False
                ind = 1
                while found_next_center is False:
                    nth_highest_index = np.argsort(abs(t_fourier))[-ind]
                    in_x = np.isin(centers[0,:], kx_total[nth_highest_index])
                    in_y = np.isin(centers[1, :], ky_total[nth_highest_index])

                    if not any(in_x & in_y) is True:
                        center = np.array([[kx_total[nth_highest_index]],[ky_total[nth_highest_index]]])
                        next_points = get_neighbors(center[0], center[1])

                        # cull all the previously measured points
                        unique_next_points = np.array([point for point in next_points[:, :, 0] if not any(
                            (np.array(np.vstack((kx_total, ky_total))).T == point).all(1))])
                        centers = np.concatenate((centers, center), axis=1)

                        if len(unique_next_points) > 0:
                            found_next_center = True

                    ind += 1

                self.k_x = unique_next_points[:, 0]
                self.k_y = unique_next_points[:, 1]

                if (n + len(self.k_x)) < self.max_modes:
                    self.added_modes.append(unique_next_points.tolist())

                if n>0:
                    self.record_intermediate_enhancement(t_fourier, kx_total, ky_total, side)

                n += len(self.k_x)



            # Measuring highest modes
            self.measure_high_modes(t_fourier, kx_total, ky_total, side)

            self.record_intermediate_enhancement(t_fourier, kx_total, ky_total, side)

            if side == 0:
                self.t_left = t_fourier
                self.k_left = np.vstack((kx_total,ky_total))

            else:
                self.t_right = t_fourier
                self.k_right = np.vstack((kx_total,ky_total))

        self.t_slm = self.compute_t(self.t_left,self.t_right,self.k_left,self.k_right)
        return self.t_slm

    def single_side_experiment(self, side):
        """Overriding the experiment class such that we can have a seperate k_x_left and k_x_right SLM side"""
        self.controller.reserve((len(self.k_x), self.phase_steps))

        phases = np.arange(self.phase_steps) / self.phase_steps * 2 * np.pi

        for n_angle in range(len(self.k_x)):
            for phase in phases:

                self.controller.slm.phases = self.get_phase_pattern(self.k_x[n_angle],
                                                                    self.k_y[n_angle],
                                                                    phase, side)
                self.controller.measure()


    def compute_t(self, t_fourier_left, t_fourier_right ,k_left, k_right):
        """
        We also need to override compute_t if we want a different kspace for left and right slm
        """
        # bepaal ruis: bahareh. Find peak & dc ofset
        t1 = np.zeros((self.controller.slm.height, self.controller.slm.width), dtype='complex128')
        t2 = np.zeros((self.controller.slm.height, self.controller.slm.width), dtype='complex128')

        for n, t in enumerate(t_fourier_left):
            phi = self.get_phase_pattern(k_left[0,n], k_left[1,n], 0, 0)
            t1 += np.exp(1j * phi) * np.conj(t)

        for n, t in enumerate(t_fourier_right):
            phi = self.get_phase_pattern(k_right[0,n], k_right[1,n], 0, 1)
            t2 += np.exp(1j * phi) * np.conj(t)

        overlap_len = int(self._overlap * self.controller.slm.width)
        overlap_begin = self.controller.slm.width // 2 - int(overlap_len / 2)
        overlap_end = self.controller.slm.width // 2 + int(overlap_len / 2)

        if self._overlap != 0:
            c = np.vdot(t2[:, overlap_begin:overlap_end], t1[:, overlap_begin:overlap_end])
            factor = c / abs(c) * np.linalg.norm(t1[:, overlap_begin:overlap_end]) / np.linalg.norm(t2[:, overlap_begin:overlap_end])
            t2 = t2 * factor

            overlap = (t1[:, overlap_begin:overlap_end] + t2[:, overlap_begin:overlap_end]) / 2
            t_full = np.concatenate([t1[:, 0:overlap_begin], overlap, t2[:, overlap_end:]], axis=1)
        else:
            t_full = np.concatenate([t1[:, 0:overlap_begin], t2[:, overlap_end:]], axis=1)


        return t_full

    def measure_high_modes(self, t_fourier, kx_total, ky_total, side):
        # Get indices of the n highest modes
        if self.high_modes == 0:
            return

        high_mode_indices = np.argsort(np.abs(t_fourier))[-self.high_modes:]

        # Store the original phase_steps
        original_phase_steps = self.phase_steps

        # Update phase_steps for high mode measurements
        self.phase_steps = self.high_phase_steps

        # Remeasure the highest modes
        for idx in high_mode_indices:
            self.k_x = np.array([kx_total[idx]])
            self.k_y = np.array([ky_total[idx]])
            self.single_side_experiment(side)
            t_fourier[idx] = self.controller.compute_transmission(self.phase_steps) / (
                        self.phase_steps / original_phase_steps)

        # Restore the original phase_steps for subsequent measurements
        self.phase_steps = original_phase_steps
        self.added_modes.append([f'Remeasuring {self.high_modes} highest modes'])

    def record_intermediate_enhancement(self, t_fourier, kx_total, ky_total, side):
        k = np.vstack((kx_total, ky_total))

        if side == 0:
            t_left = t_fourier
            k_left = k

            # Use the class-level attributes for the right side, if they're set; otherwise, initialize to zeros
            t_right = t_fourier
            k_right = np.zeros_like(k)
            overlap = self._overlap
            self._overlap = 0
        else:
            t_right = t_fourier
            k_right = k

            t_left = self.t_left  # Since side == 0 is always processed first, self.t_left should always be set by this point
            k_left = self.k_left
            overlap = self._overlap

        t_slm = self.compute_t(t_left, t_right, k_left, k_right)
        self._overlap = overlap
        self.controller.slm.phases = np.angle(t_slm)
        self.controller.slm.update()
        self.controller._source.trigger()


        self.intermediate_enhancements.append(self.controller._source.read())

    def save_experiment(self, filename="experimental_data", directory=None):
        if directory is None:
            directory = os.getcwd()  # Get current directory

        data_to_save = {
            "t_left": self.t_left,
            "t_right": self.t_right,
            "k_left": self.k_left,
            "k_right": self.k_right,
            "t_slm": self.t_slm,
            "intermediate_enhancement": self.intermediate_enhancements,
            "added_modes": self.added_modes
        }

        with open(os.path.join(directory, f'{filename}.pkl'), 'wb') as f:
            pickle.dump(data_to_save, f)


