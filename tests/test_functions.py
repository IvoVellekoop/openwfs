import numpy as np
def make_angled_wavefront(size,slope_x,slope_y):

    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    # Create the square array with specified slopes
    array = (255 * slope_y * y + 255 * slope_x * x).astype(int)
    array = array %256
    return array

def calculate_enhancement(simulation,optimised_wf):

    simulation.set_data(0)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_before = np.mean(simulation.get_image()[250:251, 250:251])

    simulation.set_data(optimised_wf)
    simulation.update()
    simulation.trigger()
    simulation.wait()
    feedback_after = np.mean(simulation.get_image()[250:251, 250:251])

    return feedback_after / feedback_before