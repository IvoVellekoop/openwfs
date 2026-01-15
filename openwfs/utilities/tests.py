import numpy as np
import astropy.units as u


def get_microscope_setup():
    """
    Convenience function returning a basic microscope (simulation) setup for testing purposes. The microscope has a static source representing a point source, and an SLM as incident field. The microscope parameters are:
    - Specimen resolution: (512, 512) pixels
    - Specimen pixel size: 100 nm
    - Numerical aperture: 0.85
    - Wavelength: 532.8 nm
    - Incident field: SLM with shape (512, 512).

    Returns:
        mic: Microscope
            The microscope object.
        slm: SLM
            The SLM object used as incident field.
        src: StaticSource
            The static source object representing the specimen.
    """
    import openwfs.simulation as owf_s

    specimen_resolution = (512, 512)
    specimen_pixel_size = 100.0 * u.nm
    numerical_aperture = 0.85
    wavelength = 532.8 * u.nm
    img_plane = np.zeros(specimen_resolution)
    img_plane[100:102, 400:402] = 1.0

    src = owf_s.StaticSource(
        data=img_plane,
        pixel_size=specimen_pixel_size,
    )

    slm = owf_s.SLM(shape=(512, 512))

    mic = owf_s.Microscope(
        src,
        magnification=1.0,
        numerical_aperture=numerical_aperture,
        wavelength=wavelength,
        incident_field=slm.field,
    )
    return mic, slm, src
