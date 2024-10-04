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

# Tests

- Tests must *not* plot figures.

# Sphinx

Common warnings:

- All line numbers are relative to the start of the docstring.
- 'WARNING: Block quote ends without a blank line; unexpected unindent'. This happens if a block of text is not properly
  wrapped and one of the lines starts with a space. To fox, remove the space at the beginning of the line.
- 'ERROR: Unexpected indentation' can be caused if a line ends with ':' and the next line is not indented or empty.