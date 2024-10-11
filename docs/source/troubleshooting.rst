.. _section-troubleshooting:

Analysis and troubleshooting
==================================================
The principles of wavefront shaping are well established and, under close-to-ideal experimental conditions, it is possible to accurately predict the signal enhancement. In practice, however, there exist many practical issues that can negatively affect the outcome of the experiment. OpenWFS has built-in functions to analyze and troubleshoot the measurements from a wavefront shaping experiment.

The ``result`` structure in :numref:`hello-wfs`, as returned by the wavefront shaping algorithm, was computed with the utility function :func:`analyze_phase_stepping`. This function extracts the transmission matrix from phase stepping measurements, and additionally computes a series of troubleshooting statistics in the form of a *fidelity*, which is a number that ranges from 0 (no sensible measurement possible) to 1 (perfect situation, optimal focus expected). These fidelities are:

* :attr:`~.WFSResults.fidelity_noise`: The fidelity reduction due to noise in the measurements.
* :attr:`~.WFSResults.fidelity_amplitude`: The fidelity reduction due to unequal illumination of the SLM.
* :attr:`~.WFSResults.fidelity_calibration`: The fidelity reduction due to imperfect phase response of the SLM.

If these fidelities are much lower than 1, this indicates a problem in the experiment, or a bug in the wavefront shaping experiment. For a comprehensive overview of the practical considerations in wavefront shaping and their effects on the fidelity, please see :cite:`Mastiani2024PracticalConsiderations`.

Further troubleshooting can be performed with the :func:`~.troubleshoot` function, which estimates the following fidelities:

* :attr:`~.WFSTroubleshootResult.fidelity_non_modulated`: The fidelity reduction due to non-modulated light., e.g. due to reflection from the front surface of the SLM.
* :attr:`~.WFSTroubleshootResult.fidelity_decorrelation`: The fidelity reduction due to decorrelation of the field during the measurement.

All fidelity estimations are combined to make an order of magnitude estimation of the expected enhancement. :func:`~.troubleshoot` returns a ``WFSTroubleshootResult`` object containing the outcome of the different tests and analyses, which can be printed to the console as a comprehensive troubleshooting report with the method :meth:`~.WFSTroubleshootResult.report()`. See ``examples/troubleshooter_demo.py`` for an example of how to use the automatic troubleshooter.

Lastly, the :func:`~.troubleshoot` function computes several image frame metrics such as the *unbiased contrast to noise ratio* and *unbiased contrast enhancement*. These metrics are especially useful for scenarios where the contrast is expected to improve due to wavefront shaping, such as in multi-photon excitation fluorescence (multi-PEF) microscopy. Furthermore, :func:`~.troubleshoot` tests the image capturing repeatability and runs a stability test by capturing and comparing many frames over a longer period of time.
