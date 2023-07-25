import numpy as np
def make_angled_wavefront(size, slope_x, slope_y):
    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, size), np.linspace(-np.pi, np.pi, size))

    # Create the square array with specified slopes
    array = (slope_y * y + slope_x * x)
    array = 2 * np.pi * (array - array.min()) / (array.max() - array.min()) - np.pi

    return array

def calculate_enhancement(simulation,optimised_wf):

    simulation.set_data(0)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_before = np.mean(simulation.read()[250, 250])

    simulation.set_data(optimised_wf)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_after = np.mean(simulation.read()[250, 250])

    return feedback_after / feedback_before