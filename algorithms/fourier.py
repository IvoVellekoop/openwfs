import math
import numpy as np
from typing import Any, Annotated


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
        k_set: the [kx,ky] matrix 2 x n.
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

        x = np.arange(start, end, 1 / ((width - overlap) + overlap))[np.newaxis, :]
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
            c = np.vdot(t2[:, :overlap_len], t1[:, -overlap_len:])
            factor = c / abs(c) * np.linalg.norm(t2[:, :overlap_len]) / np.linalg.norm(t1[:, -overlap_len:])
            t2 = t2 / factor

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

