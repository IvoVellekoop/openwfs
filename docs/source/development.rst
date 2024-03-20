OpenWFS Development
====================

Running the tests and examples
--------------------------------------------------
To download the source code, including tests and examples, clone the repository from GitHub. OpenWFS uses `poetry` :cite:`Poetry` for package management, so you have to download and install Poetry first. Then, navigate to the location where you want to store the source code, and execute the following commands to clone the repository, set up the poetry environment, and run the tests.

.. code-block:: shell

    git clone https://github.com/IvoVellekoop/openwfs/
    cd openwfs`
    poetry install --with dev --with docs
    poetry run pytest


Building the documentation
--------------------------------------------------

.. only:: html or markdown
    The html, and pdf versions of the documentation, as well as the `README.md` file in the root directory of the repository, are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

.. only:: latex
    The html version of the documentation, as well as the `README.md` file in the root directory of the repository, and the pdf document you are currently reading are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

    To build the documentation, make sure to have install the required packages first using `poetry install --with docs`. Then, run the following commands to build the html and pdf versions of the documentation, and to auto-generate `README.md`.

    Note that for building the pdf version of the documentation, you need to have `xelatex` installed, which comes with the MiKTeX distribution of LaTeX :cite:`MiKTeX`.

.. code-block:: shell

    poetry shell
    cd docs
    make clean
    make html
    make markdown
    make latex
    cd _build/latex
    xelatex openwfs
    xelatex openwfs


Reporting bugs and contributing
--------------------------------------------------
TODO


Implementing new devices
--------------------------------------------------

Internals of the synchronization mechanism
+++++++++++++++++++++++++++++++++++++++++
For the synchronization, `Device` keeps a global state which can be either
- `moving = True`. One or more actuators may be busy. No measurements can be made (none of the detectors is busy).
- `moving = False` (the 'measuring' state). One or more detectors may be busy. All actuators must remain static (none of the actuators is busy).

When an actuator is started, or when a detector is triggered, it calls `self._start` to request a switch to the correct global state. If a state switch is needed, this function blocks until all devices of the other type are ready. For example, if a detector calls `_start`, the framework waits for all actuators to finish, before the switch is made. For detectors and processors, `_start` is called automatically by `trigger()`, so there is never a need to call it explicitly.


Implementing a detector
++++++++++++++++++++++++++++++++++
To implement a detector, the user should subclass the `Detector` base class, and implement whatever properties and logic appropriate to the device. In particular, the user should implement the `_do_trigger` method to start the measurement process, and the  `~Detector._fetch()` method to fetch the data from the hardware, optionally process it, and return it as a numpy array.

If `duration`, `pixel_size` and `data_shape` are known, they should be passed to the base class constructor. If these properties may change during operation, the user should override the `duration`, `pixel_size` and `data_shape` properties to provide the correct values dynamically. If the `duration` is not known in advance (for example, when waiting for a hardware trigger), the Detector should implement the `busy` function to poll the hardware for the busy state.

If the detector was created with the flag `multi_threaded = True`, then `_fetch` will be called from a worker thread. This way, the rest of the program does not need to wait for transferring data from the hardware, or for computationally expensive processing tasks. OpenWFS automatically prevents any modification of public properties between the calls to `_do_trigger` and `_fetch`, which means that the `_fetch` function can safely read (not write) these properties without the chance of a race condition. Care must be taken, however, not to read or write private fields from `_fetch`, since this is not thread-safe.

Implementing a processor
++++++++++++++++++++++++++++++++++
To implement a custom processor, derive from the `Processor` base class. Override the `__init__` function to pass all sources to the base class constructor. In addition, implement the `_fetch` method to process the data. The framework will wait until the data from all sources is available, and pass this data as input arguments to the `~Detector._fetch()` function. See `Microscope._fetch`, or any other `Processor` object for an example of how to implement this function.

Implementing an actuator
+++++++++++++++++++++++++++++++
To implement an actuator, the user should subclass the `Actuator` base class, and implement whatever properties and logic appropriate to the device. All methods that start the actuator (`update()`, `move()` or similar), should first call  `self._start()` to request a state switch to the `moving` state. Otherwise, no special logic is required for the actuator to be compatible with OpenWFS.



