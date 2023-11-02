import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def angular_difference(A, B):
    np.mod(A - B, 2 * np.pi) - np.pi


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


def plot_dense_grid(x, y, data):
    # TODO: make unused pixels NaN
    """
    Plot a dense grid visualization for given x, y coordinates and complex data.
    Grid points not in the provided data are set to zero (white).

    Parameters:
    - x, y: Arrays of x and y coordinates.
    - data: Array of complex data corresponding to x, y coordinates.
    """

    # Create the dense grid based on the range of x and y values
    x_range = np.arange(np.min(x), np.max(x) + 1)
    y_range = np.arange(np.min(y), np.max(y) + 1)
    dense_grid = np.full((len(y_range), len(x_range)), -1, dtype=complex)  # initialize with -1

    # Map the real data points onto the dense grid
    dense_grid[y, x] = data

    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    dense_grid = np.zeros((y_max - y_min, x_max - x_min)) + np.nan
    for i in range(len(data)):
        dense_grid[y[i], x[i]] = data[i]

    for xi in x_range:
        for yi in y_range:
            if (xi, yi) in zip(x, y):
                index_data = np.where((x == xi) & (y == yi))
                dense_grid[np.where(y_range == yi), np.where(x_range == xi)] = data[index_data]

    # Setting values of -1 to 0 (white)
    dense_grid[dense_grid == -1] = 0

    # Custom colormap with zero values set to white
    colors = [(1, 1, 1)] + [plt.cm.viridis(i) for i in range(1, 256)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list('new_cmap', colors, N=256)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(dense_grid) / np.max(np.abs(dense_grid)), extent=(np.min(x) - 0.5, np.max(x) + 0.5,
                                                                        np.min(y) - 0.5, np.max(y) + 0.5),
               origin='lower', cmap=new_cmap, aspect='auto')
    plt.colorbar(label='Normalised amplitude')
    plt.clim([0, 1])
    plt.xlabel(r'K$_x$')
    plt.ylabel(r'K$_y$')


def plot_dense_grid_no_empty_spaces(x, y, data):
    """
    Plot a dense grid visualization for given x, y coordinates and complex data.
    Grid points not in the provided data are set to zero (white).

    Parameters:
    - x, y: Arrays of x and y coordinates.
    - data: Array of complex data corresponding to x, y coordinates.
    """

    # Create the dense grid based on the range of x and y values
    x_range = np.arange(np.min(x), np.max(x) + 1)
    y_range = np.arange(np.min(y), np.max(y) + 1)
    dense_grid = np.full((len(y_range), len(x_range)), -1, dtype=complex)  # initialize with -1

    # Map the real data points onto the dense grid
    for xi in x_range:
        for yi in y_range:
            if (xi, yi) in zip(x, y):
                index_data = np.where((x == xi) & (y == yi))
                dense_grid[np.where(y_range == yi), np.where(x_range == xi)] = data[index_data]

    # Setting values of -1 to 0 (white)
    dense_grid[dense_grid == -1] = 0

    # Custom colormap with zero values set to white
    colors = [plt.cm.viridis(i) for i in range(1, 256)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list('new_cmap', colors, N=256)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(dense_grid) / np.max(np.abs(dense_grid)), extent=(np.min(x) - 0.5, np.max(x) + 0.5,
                                                                        np.min(y) - 0.5, np.max(y) + 0.5),
               origin='lower', cmap=new_cmap, aspect='auto')
    plt.colorbar(label='Normalised amplitude')
    plt.clim([0, 1])
    plt.xlabel(r'K$_x$')
    plt.ylabel(r'K$_y$')
