import numpy as np
from scipy.ndimage import binary_dilation as dilate
from scipy.ndimage import binary_erosion as erode
import matplotlib.pyplot as plt


def pathfinding(fake_data_norm, Nx, Ny, fig=None, axs=None):
    """
    pathfinding algorithm
    algorithm for selecting modes for FourierWFS to increase efficiency
    """
    # Initialize k-space measurement array with plenty large boundaries
    neighbours = np.ones(shape=[3, 3])          # Boolean array, morph. structering element.
    measurements = np.zeros([Nx, Ny], dtype=np.complex128)  # Complex array, stores the measurements
    is_measured = np.full([Nx, Ny], False)      # Boolean array, which modes have been measured
    is_selected = np.full([Nx, Ny], False)      # Boolean array, modes selected for measurement
    is_edge_mode = np.full([Nx, Ny], False)     # Bool array of edge modes


    # Initialize target mode indices
    tx = int((Nx-1) / 2)
    ty = int((Ny-1) / 2)


    while not is_criterion_reached(measurements):
        # Select modes to measure
        target_neighbourhood = dilate(true_at(tx, ty, Nx, Ny), neighbours)
        is_selected = np.logical_and(target_neighbourhood, np.logical_not(is_measured))

        # Plot
        if fig is not None and axs is not None:
            axs[0].clear()
            axs[0].imshow(np.abs(measurements), vmin=0, vmax=1)
            axs[0].plot(ty, tx, 'or')
            axs[0].set_title('Measurements')

            img = np.stack((is_selected*255, is_edge_mode*255, is_measured*255), axis=2)

            axs[1].clear()
            axs[1].imshow(img)
            axs[1].plot(ty, tx, 'or')
            axs[1].set_title('Boolean arrays')

            plt.draw()
            plt.pause(0.5)

        # Measure modes
        selected_indices = np.argwhere(is_selected)                     # Indices of selected modes
        for sx, sy in selected_indices:                                 # Loop selected indices
            measurements[sx, sy] = measure_mode(sx, sy, fake_data_norm)         # Measure mode

            # Plot
            if fig is not None and axs is not None:
                axs[0].clear()
                axs[0].imshow(np.abs(measurements), vmin=0, vmax=1)
                axs[0].plot(ty, tx, 'or')
                axs[0].set_title('Measurements')

                axs[1].plot(sy, sx, 'ok')

                axs[1].set_title(f'{sx}, {sy}')
                plt.draw()
                plt.pause(0.5)

        is_measured = np.logical_or(is_measured, is_selected)           # Update is_measured

        # Find highest edge mode
        is_edge_mode = edge(is_measured, neighbours)                    # Bool array of edge modes
        edge_measurements = np.where(is_edge_mode, measurements, 0)     # Numeric array

        tx, ty = argmax2d(np.abs(edge_measurements))                    # Highest edge mode


def edge(A, neighbours):
    return np.logical_and(A, np.logical_not(erode(A, neighbours)))

def true_at(tx, ty, Nx, Ny):
    z = np.zeros(shape=[Nx, Ny])
    z[tx, ty] = True
    return z

def argmax2d(A):
    return np.unravel_index(A.argmax(), A.shape)


def measure_mode(sx, sy, fake_data_norm):
    return fake_data_norm[sx, sy]

def is_criterion_reached(measurements):
    return np.abs(measurements).sum() > 100


Nx = 21
Ny = 21

# === Fake data === #
kmax = 500
kx = np.linspace(-kmax, kmax, Nx).reshape(1, Nx)
ky = np.linspace(-kmax, kmax, Ny).reshape(Ny, 1)

kr = np.sqrt(kx**2 + ky**2)
fake_ftdata = np.exp(1j * np.random.rand(Nx, Ny) * 2 * np.pi) / (1 + kr)
fake_data = np.fft.ifft2(np.fft.fftshift(fake_ftdata)) * np.exp(-kr**2 / 5e5)
fake_data_norm = fake_data / np.max(np.abs(fake_data))

# fake_data_norm = np.array(((1,2,3,4,5), (6,7,8,9,10), (11,12,13,14,15), (16,17,18,19,20), (21,22,23,24,25)))

plt.figure()
plt.imshow(np.abs(fake_data_norm), vmin=0, vmax=1)
plt.title('Fake data k-space')
plt.colorbar()
plt.draw()
plt.pause(1)
# ================= #


fig, axs = plt.subplots(1, 2, figsize=(10, 7))

pathfinding(fake_data_norm, Nx, Ny, fig=fig, axs=axs)

plt.title('Reached criterion')
plt.show()


# === Function descriptions === #
# zeros(shape)                          Return numeric array of requested shape, filled with 0
# false(shape)                          Return boolean array of requested shape, filled with false
# measure_mode(tx, ty)                  Run phase stepping measurement of this mode and return the value
# erode(boolean_array, struct_elem)     Morphological erode operation on boolean_array, with struct_elem.
# dilate(boolean_array, struct_elem)    Morphological dilate operation on boolean_array, with struct_elem.
# neighbours                            Structering element array: true(shape=[3, 3])
# argmax(array)                         Return index of maximum of array. numpy.argmax
# not, and                              Element-wise boolean operations on boolean arrays.
#                                       numpy.logical_not, numpy.logical_and
# where(condition, array1, array2)      For each element, if true select from array 1, else from
#                                       array 2. numpy.where.


