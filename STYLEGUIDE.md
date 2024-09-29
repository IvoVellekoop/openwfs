# Documentation

- General documentation is written in reStructuredText format, and stored in the `docs` directory.
- All public functions and classes should have docstrings, written in the Google style.
- Input arguments and return values should be type hinted in the function header only, not in the docstring.

# General

- the package `black` is used to ensure correct formatting. Install with `pip install black` and run in the terminal using `black .` when located at the root of the repository. 

# Tests

- Tests must *not* plot figures.