# Changelog

Below are curated lists of major (breaking or behavior-changing) and minor (backwards-compatible fixes and improvements)
changes based on the Git history since tag `v1.0.0`.

## [1.1.0] -November 2025

#### Major changes

- Feature: Added support for Zaber stages
- Feature: Implemented HDR camera wrapper, wraps any given camera and makes it work in High Dynamic Range (HDR) mode.
- Behavior/API: Camera exposure naming normalized ("exposure_time" → "exposure").
- Behavior change: Camera now chooses highest bit depth by default.
- Platform support: Python 3.9 no longer supported. Minimum version is now 3.10.
- Importing subpackages now automatic, e.g. `from openwfs.devices import ZaberLinearStage`

#### Minor changes and improvements

- BREAKING: Simplified `utilities.propagation` pattern generator (e25705e, 2025-03-20)
- Camera/GenICam robustness and features
    - Added a rudimentary live viewer (d71632a, 2025-10-07)
    - Made GenICam `Camera` less sensitive to missing nodes (fcee0bd, 2025-03-20)
- Devices, simulation, and stages
    - Added linear Z-stage (cdc6aa5, 2025-03-20)
    - Add defocus from ZStage position into simulated image (5face19, 2025-03-19)
    - Added single-stage mock + unit tests (89660ee, 2025-01-29; ffb781d UT; 7bb205f docs)
- SLM and OpenGL windowing
    - Fixed OpenCL transform matrix and windowed SLM creation (75a8218, 5498133, 2024-12-12)
    - Full-screen SLM windows now close automatically when the monitor is disconnected (instead of taking over the
      primary monitor!); improved error handling (96484e4, 498e822, 2024-12-10)
- Packaging and dependencies
    - The version of OpenWFS can now be queried using `openwfs.__version__`, which returns the package version, or `openwfs.version()` which also includes the commit hash if commits were made after the version release tag.
    - Astropy requirement adjustments: raised then reduced due to Python 3.10 compatibility with astropy 7 (e4c0928 →
      964e6c9, 2025-10-28)
- Documentation & README
    - Various documentation updates
    - Added CHANGELOG
    - Added a deployment checklist
