from openwfs.simulation import SimulatedWFS
import numpy as np
from openwfs.algorithms import CharacterisingFDR
from openwfs.feedback import Controller, SingleRoi
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append('..//')
from functions import angular_difference


def show_pathfinding_fourier_stepping():
    """
    Performs a Fourier-based pathfinding experiment

    ToDo: Plot intermediate results
    """
    sim = SimulatedWFS(width=512, height=512, beam_profile_fwhm=300)

    roi_detector = SingleRoi(sim, x=256, y=256, radius=0)
    roi_detector.trigger()
    correct_wf = (np.load("..//data/fourier/optimised_wf.npy") / 255) * 2 * np.pi - np.pi
    sim.set_ideal_wf(correct_wf)

    plt.imshow((correct_wf % (2 * np.pi)) - np.pi)
    plt.colorbar(label='Phase offset (radians)')

    matplotlib.rcParams.update({'font.size': 16})

    controller = Controller(detector=roi_detector, slm=sim)
    alg_char = CharacterisingFDR(phase_steps=3, overlap=0.1, max_modes=50, high_modes=0, high_phase_steps=17,
                                 intermediates=True, controller=controller)
    alg_char.execute().t

    pass


print(show_pathfinding_fourier_stepping())
