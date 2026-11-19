import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from openwfs.utilities.patterns import zernike, coordinate_range, noll_to_nm
from openwfs.utilities.utilities import unitless

# example that plots the first 15 Zernike modes using Noll indexing in a triangular grid layout,
# with the number of modes in each row increasing from 1 to 5.

shape = (256, 256)
extent = (2.0, 2.0)

# Create the pupil mask once.
y, x = coordinate_range(shape, extent)
pupil = unitless(np.hypot(x, y)) <= 1

fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(5, 9, hspace=0.25, wspace=0.1)

indices = [1, 2, 3, 5, 4, 6, 9, 7, 8, 10, 15, 13, 11, 12, 14]  # order of Noll indices to plot
j_idx = 0
for row in range(5):
    # Center rows containing 1, 2, 3, 4 and 5 modes.
    columns = np.arange(row + 1) * 2 + (4 - row)

    for column in columns:
        j = indices[j_idx]
        mode = zernike(shape, j, extent)
        mode = np.ma.masked_where(~pupil, mode)

        ax = fig.add_subplot(grid[row, column])
        ax.imshow(
            mode,
            cmap=cc.cm.rainbow_bgyr_10_90_c83,
            origin="lower",
        )
        # set title with n,m index
        ax.set_title(f"$j={j}$, $(n,m)={noll_to_nm(j)}$")
        # set title size to 10
        ax.title.set_fontsize(9)
        ax.axis("off")

        j_idx += 1

fig.suptitle("First 15 Zernike modes, Noll indexing")
plt.show()
