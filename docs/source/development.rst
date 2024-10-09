.. _section-development:

OpenWFS Development
==============================================

Running the tests and examples
--------------------------------------------------
To download the source code, including tests and examples, clone the repository from GitHub :cite:`openwfsgithub`. OpenWFS uses ``poetry`` :cite:`Poetry` for package management, so you have to download and install Poetry first. Then, navigate to the location where you want to store the source code, and execute the following commands to clone the repository, set up the poetry environment, and run the tests.

.. code-block:: shell

    git clone https://github.com/IvoVellekoop/openwfs/
    cd openwfs
    poetry install --with dev --with docs
    poetry run pytest

The examples are located in the ``examples`` directory. Note that a lot of functionality is also demonstrated in the automatic tests located in the ``tests`` directory. As an alternative to downloading the source code, the samples can also be copied directly from the example gallery on the documentation website :cite:`readthedocsOpenWFS`. 

Important to note for adding hardware devices, is that many of the components rely on third-party, in some case proprietary drivers. For using NI DAQ components, the nidaqmx package needs to be installed, and for openCV and Genicam their respective drivers need to be installed. The specific requirements are always listed in the documentation of the functions and classes that require packages like these.

Building the documentation
--------------------------------------------------

.. only:: html or markdown

    The html, and pdf versions of the documentation, as well as the ``README.md`` file in the root directory of the repository, are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

.. only:: latex

    The html version of the documentation, as well as the ``README.md`` file in the root directory of the repository, and the pdf document you are currently reading are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

Note that for building the pdf version of the documentation, you need to have ``xelatex`` installed, which comes with the MiKTeX distribution of LaTeX :cite:`MiKTeX`. Then, run the following commands to build the html and pdf versions of the documentation, and to auto-generate ``README.md``.

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
Bugs can be reported through the GitHub issue tracking system. Better than reporting bugs, we encourage users to *contribute bug fixes, new algorithms, device drivers, and other improvements*. These contributions can be made in the form of a pull request :cite:`zandonellaMassiddaOpenScience2022`, which will be reviewed by the development team and integrated into the package when appropriate. Please contact the current development team through GitHub :cite:`openwfsgithub` to coordinate such contributions.


Implementing new devices
--------------------------------------------------
To implement a custom device (actuator, detector, processor), it is important to first understand the implementation of the mechanism that synchronizes detectors and actuators. To implement this mechanism, the :class:`~.Device` class keeps a global state which can be either

    - ``moving = True``. One or more actuators may be busy. No measurements can be made (none of the detectors is busy).
    - ``moving = False`` (the 'measuring' state). One or more detectors may be busy. All actuators must remain static (none of the actuators is busy).

When an actuator is started, or when a detector is triggered, it calls ``self._start`` to request a switch to the correct global state. If a state switch is needed, this function blocks until all devices of the other device type are ready. For example, if an actuator calls ``_start``, the framework waits for all detectors to complete their measurements (up to latency, see :numref:`device-synchronization`) before the switch is made. Note that for  detectors and processors, ``_start`` is called automatically by :meth:`~.Device.trigger()`, so there is never a need to call it explicitly.


Implementing a detector
++++++++++++++++++++++++++++++++++
To implement a detector, the user should subclass the :meth:`~.Detector` base class, and implement properties and logic to control the detector hardware. In particular, the user should implement the :meth:`~Detector._do_trigger` method to start the measurement process in the hardware if needed, and the  :meth:`~Detector._fetch()` method to fetch the data from the hardware, optionally process it, and return it as a numpy array. A simple example of a detector that can be used as a starting point is the :class:`mockdevices.NoiseDetector`, which generates random noise with a given shape and pixel size.

If ``duration``, ``pixel_size`` and ``data_shape`` are constants, they should be passed to the base class constructor. If these properties may change during operation, the user should override the ``duration``, ``pixel_size`` and ``data_shape`` properties to provide the correct values dynamically. If the ``duration`` is not known in advance (for example, when waiting for a hardware trigger), the Detector should implement the ``busy`` function to poll the hardware for the busy state.

If the detector is created with the flag ``multi_threaded = True``, then :meth:`~Detector._fetch()` will be called from a worker thread. This way, the rest of the program does not need to wait for transferring data from the hardware, or for computationally expensive processing tasks. OpenWFS automatically prevents any modification of public properties between the calls to :meth:`~Detector._do_trigger` and :meth:`~Detector._fetch`, which means that the ``_fetch`` function can safely read (not write) these properties without the chance of a race condition. Care must be taken, however, not to read or write private fields from ``_fetch``, since this is not thread-safe.


Implementing a processor
++++++++++++++++++++++++++++++++++
To implement a data processing step that dynamically processes data from one or more input detectors, implement a custom processor. This is done by deriving from the :class:`~.Processor` base class and implementing the ``__init__`` function. This function should pass a list of all upstream nodes, i.e. all detectors which provide the input signals to the processor, the base class constructor. In addition, the :meth:`~Detector._fetch()` method should be implemented to process the data. The framework will wait until the data from all sources is available, and calls :meth:`~.Detector._fetch()` with this data as input. See the implementation of :class:`~.GaussianNoise` or any other processor for an example of how to implement this function.

Implementing an actuator
+++++++++++++++++++++++++++++++
To implement an actuator, the user should subclass the :class:`~Actuator` base class, and implement whatever properties and logic appropriate to the device. All methods that start the actuator (e.g. ``update()`` or ``move()``), should first call  ``self._start()`` to request a state switch to the ``moving`` state. As for detectors, actuators should either specify a static ``duration` and ``latency`` if known, or override these properties to return run-time values for the duration and latency. Similarly, if the duration of an action of the actuator is not known in advance, the class should override ``busy`` to poll for the action to complete.

Implementing new algorithms
--------------------------------------------------
The algorithms that are included in OpenWFS are wrapped in classes with two common attribute: ``slm``, ``feedback``, which respectively hold a :class:`~.PhaseSLM` object to control the SLM and a :class:`~Detector` object that returns the feedback signals used in the optimization. For algorithms that support optimizing multiple targets simulaneously, the ``feedback`` detector may return an array of values.
In addition, all algorithms have an ``execute()`` method that executes the algoritm and returns the measured transmission matrix, along with statistics about the measurements in a :class:`WFSResults` structure (see :numref:`section-troubleshooting).
When implementing a new algorithm, it is perfectly acceptable to deviate from this convention. However, if an algorithm follows the convention described above, it can directly be wrapped in a `WFSController` so that it can be used in Micro-Manager (see :numref:`section-micromanager`)
As can be seen in the example in :numref:`hello-wfs`,  OpenWFS abstracts all hardware interactions in the calls to  ``slm.set_phases`` and ``feedback.trigger``.




