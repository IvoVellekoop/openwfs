Conclusion
====================

In this work we presented an open-source Python package for conducting and simulating wavefront shaping experiments. The focus is on a modular design that encapsulates the details of low-level hardware control so that the developer can focus on the design of wavefront shaping algorithms and the high-level experimental control.

OpenWFS incorporates features to reduce the chances of errors in the design of wavefront shaping code. Notably, the use of units of measure prevents the accidental mixing of units, and the automatic synchronization mechanism ensures that hardware is properly synchronized without the need to write any synchronization code. Finally, the ability to simulate full experiments, and to mock individual components, allows the user to test wavefront shaping algorithms without the need for physical hardware. We find this feature particularly useful since there is a lot that can go wrong in an experiment, (also see :cite:`Mastiani2024PracticalConsiderations`), and experimental issues and software issues are not always easy to distinguish. With OpenWFS, it is now possible to fully test the algorithms before entering the lab, which can save a lot of time and frustration.

We envision that OpenWFS will hold a growing collection of components for hardware control, advanced simulations, and wavefront shaping. Further expansion of the supported hardware is of high priority, especially wrapping c-based libraries and adding support for  Micro-Manager device adapters. The standardised interfaces for detectors, actuators and SLMs will greatly simplify developing reusable code that can be used across different setups and experiments. The simulation tools may additionally be used for research and education, ushering in a new phase of applications in wavefront shaping. We therefore encourage the reader to join us in developing new algorithms and components for this framework.

Code availability
------------------------------------------------
The code for OpenWFS is available on GitHub :cite:`openwfsgithub`, it is licensed under the BSD 3-Clause license. When using OpenWFS in your work, please cite this current paper. Examples and documentation for this project are available at `Read the Docs <https://openwfs.readthedocs.io/en/latest/>`_ :cite:`openwfsdocumentation`.

%endmatter%




