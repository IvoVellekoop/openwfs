import numpy as np
import astropy.units as u


def get_test_microscope(
    slm_args={},
    src_args={},
    mic_args={},
):
    """
    Convenience function returning a basic microscope (simulation) setup for testing purposes. The microscope has a static source representing a point source, and an SLM as incident field. The default settings are:
    - Specimen resolution: (512, 512) pixels
    - Specimen pixel size: 100 nm
    - Numerical aperture: 0.85
    - Wavelength: 532.8 nm
    - Incident field: SLM with shape (512, 512).
    - Magnification: 1

    Any setting can be customized by passing the desired arguments to the function. The arguments are passed as dictionaries, which are then passed to the respective openwfs function. Three dictionaries can be used as input which control the arguments to the StaticSource, SLM, and Microscope. The keys of the dictionaries are the variable name which are passed to the StaticSource, SLM and Microscope construtor.

    For example, to change the numerical aperture to 0.95, you can call the function as follows:
    ```python
    mic, slm, src = get_test_microscope(mic_args={"numerical_aperture": 0.95})
    mic, slm, src = get_test_microscope(slm_args={"shape": (256, 256)})
    mic, slm, src = get_test_microscope(src_args={"pixel_size": 50 * u.nm})
    ```

    Inputs:
        slm_args: dict
            Dictionary containing the arguments for the SLM constructor.
        src_args: dict
            Dictionary containing the arguments for the StaticSource constructor.
        mic_args: dict
            Dictionary containing the arguments for the Microscope constructor.
    Returns:
        mic: Microscope
            The microscope object.
        slm: SLM
            The SLM object used as incident field.
        src: StaticSource
            The static source object representing the specimen.
    """
    import openwfs.simulation as owf_s

    default_slm_args = {
        "shape": (512, 512),
    }
    slm_args = default_slm_args | slm_args

    src_data = np.zeros((512, 512))
    src_data[256, 256] = 1.0
    default_src_args = {
        "data": src_data,
        "pixel_size": 100 * u.nm,
    }
    src_args = default_src_args | src_args

    src = owf_s.StaticSource(**src_args)

    slm = owf_s.SLM(**slm_args)

    default_mic_args = {
        "magnification": 1,
        "numerical_aperture": 0.85,
        "wavelength": 532.8 * u.nm,
        "incident_field": slm.field,
    }
    mic_args = default_mic_args | mic_args

    mic = owf_s.Microscope(src, **mic_args)
    return mic, slm, src
