import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def calculate_enhancement(simulation, optimised_wf, x=256, y=256):
    simulation.set_data(0)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_before = np.mean(simulation.read()[x, y])

    simulation.set_data(optimised_wf)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_after = np.mean(simulation.read()[x, y])

    return feedback_after / feedback_before


def measure_feedback(simulation, optimised_wf, x=256, y=256):
    simulation.set_data(optimised_wf)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    return np.mean(simulation.read()[x, y])

