======================
Coordinates and units
======================
Throughout OpenWFS, quantities with physical dimensions are represented as an `astropy.units.Quantity`, which holds metadata about the unit of measure. This allows for calculations to be performed with correct units, and for automatic unit conversion where necessary. Importantly, it prevents errors caused by passing a quantity in incorrect units, such as passing a wavelength in micrometers when the function expects a wavelength in nanometers. Units are converted automatically, so one may for example specify a time in microseconds, milliseconds, or seconds.


In addition, OpenWFS attaches metadata to data arrays using the functions `set_pixel_size`. Pixel sizes can represent a physical length (e.g. as in the size pixels on an image sensor), or other units such as time (e.g. as the sampling period in a time series). OpenWFS fully supports anisotropic pixels, where the pixel size in the x and y directions are different (but still need to have the same unit). The pixel size can be retrieved using the `get_pixel_size` function.

As an alternative accessing the pixel size directly `get_extent` provide access to the extent of the array, which is always equal to the pixel size times the shape of the array. The extent can be set directly using `set_extent`, which internally converts the extent to pixel size and updates the pixel size metadata on the array.

For two-dimensional data sets, the width and height are stored as the second and first dimension: `(height, width)`. This convention is consistent with the way matplotlib displays numpy arrays and with the idea that the first index is the row index and the second index is the column index.

To obtain a vector with  coordinates for a data array with pixel size metadata set, use the `get_coordinates` function, which takes an array and axis as input and returns a vector of coordinates along that dimension of the array. By default, coordinates have the origin at the center of the array, and they represent the coordinates of the _centers_ of the pixels. For example, if an array has pixel size 1.0 and shape (3,), the `extent` of the array is 3.0, and the coordinates will be `[-1, 0, 1]`, where the pixels range from -1.5 to -0.5, -0.5 to 0.5, and 0.5 to 1.5, respectively.

In OpenWFS, three types of units are used:
    - Length units, such as meters, millimeters, or micrometers
    - Time units, such as seconds or milliseconds
    - Pupil units, which are dimensionless and represent the pupil plane of the microscope objective in a microscope. The coordinates are normalized so that the pupil plane extends from -1 to 1 in both dimensions, which is consistent with the convention used in Zernike polynomials.



