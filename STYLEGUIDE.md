# Documentation

- General documentation is written in reStructuredText format, and stored in the `docs` directory.
- All public functions and classes should have docstrings, written in the Google style.
- Input arguments and return values should be type hinted in the function header only, not in the docstring.

# General

- The package `black` is used to ensure correct formatting.
  When using PyCharm, just install black through the settings dialog.
- PyCharm warnings and errors should be fixed. Exceptions:
    - PEP 8: E501 line too long. May be disabled. This is already checked by black. For docstrings, keeping a string
      line limit can be very cumbersome.
    - PEP 8:E203 whitespace before ':'. May be disabled. This is already checked by (and conflicts with) black.

# Layout

- Line length of code is limited to 120 characters by `black`.
- Use soft wrapping for Markdown (`.md`) and reStructuredTest (`.rst`) files.

# Tests

- Tests must *not* plot figures.

# Sphinx

Common warnings:

- All line numbers are relative to the start of the docstring.
- 'WARNING: Block quote ends without a blank line; unexpected unindent'. This happens if a block of text is not properly
  wrapped and one of the lines starts with a space. To fox, remove the space at the beginning of the line.
- 'ERROR: Unexpected indentation' can be caused if a line ends with ':' and the next line is not indented or empty.

# Indexing

- For 2-D data, the first index is the row, the second index is the column.
- Multi-dimensional data sets should be indexed such that the first index corresponds
  to the slowest changing dimension, and the last index corresponds to the fastest changing dimension. For example, a
  sequence of camera frames should be indexed as `frames[frame_index, row, column]`.
- If you find yourself needing to index a 3D array with `array[:, :, 0]`, using a lot of `expand_dims` or `reshape`
  commands, this is an indication that the indexing is not optimal, both in terms of performance and readability.
- Don't use ``[i,...]`` for indexing, use ``[i]`` instead.
- Prefer ``len(x)`` over ``x.shape(0)`` if the leading dimension represents a different 'type'. So, for a list of
  images, use ``len(images)`` instead of ``images.shape(0)``. But to access the number of rows in an image, use
  ``image.shape(0)``.

  # Properties
- Document properties in the getter method only, not in the setter method. Also describe what happens if the property is
  set.
- 