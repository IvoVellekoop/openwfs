# Documentation

- General documentation is written in reStructuredText format, and stored in the `docs` directory.
- All public functions and classes should have docstrings, written in the Google style.
- Input arguments and return values should be type hinted in the function header only, not in the docstring.

# General

- all .py files MUST be formatted with the 'black' autoformatter. This can be done by installing the 'black' package,
  and running `black .` in the root directory. black is automatically installed when the development dependencies are
  included.

# Tests

- Tests must *not* plot figures.