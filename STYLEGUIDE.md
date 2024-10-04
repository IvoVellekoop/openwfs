# Documentation

- General documentation is written in reStructuredText format, and stored in the `docs` directory.
- All public functions and classes should have docstrings, written in the Google style.
- Input arguments and return values should be type hinted in the function header only, not in the docstring.

# General

- The package `black` is used to ensure correct formatting. Install with `pip install black` and run in the terminal
  using `black .` when located at the root of the repository.

# Tests

- Tests must *not* plot figures.

# Sphinx

Common warnings:

- All line numbers are relative to the start of the docstring.
- 'WARNING: Block quote ends without a blank line; unexpected unindent'. This happens if a block of text is not properly
  wrapped and one of the lines starts with a space. To fox, remove the space at the beginning of the line.
- 'ERROR: Unexpected indentation' can be caused if a line ends with ':' and the next line is not indented or empty.