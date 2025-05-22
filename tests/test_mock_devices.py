import astropy.units as u
import numpy as np
from astropy.units import Quantity

from ..openwfs.processors import HDRCamera
from ..openwfs.simulation import StaticSource, SLM, Stage, Camera


def test_mock_slm():
    # TODO: at some point, merge this with the test_slm.py file
    # create a mock SLM object with a given shape
    slm = SLM(shape=(480, 640))

    # clear the slm
    slm.set_phases(0)

    # test if all elements of slm.phases are 0
    assert np.all(slm.phases.read() == 0)

    # set a pattern, don't update yet
    pattern = np.random.uniform(0.0, 1.9 * np.pi, size=(48, 64))
    slm.set_phases(pattern, update=False)

    # test if all elements of slm.phases are still 0
    assert np.all(slm.phases.read() == 0)

    # update the slm, and read back
    slm.update()
    scaled_pattern = np.repeat(np.repeat(pattern, 10, axis=0), 10, axis=1)
    assert np.allclose(slm.phases.read(), scaled_pattern, atol=1.0 * np.pi / 256)


def test_single_stage():
    stage = Stage("x", Quantity(10, u.um))
    assert stage.position == 0
    assert stage.axis == "x"

    stage.position = Quantity(5, u.um)
    assert stage.position == Quantity(0, u.um)

    stage.position = Quantity(15, u.um)
    assert stage.position == Quantity(20, u.um)

    stage.position = Quantity(-5, u.um)
    assert stage.position == Quantity(0, u.um)

    stage.position = Quantity(-15, u.um)
    assert stage.position == Quantity(-20, u.um)


def test_mock_camera():
    """Tests a simulated camera and HDR processing"""

    # create a mock camera object with a static image
    img = np.random.uniform(0, 2.0, size=(48, 64))
    img[0, 0] = 0.9813272622363753
    src = StaticSource(img, pixel_size=50 * u.nm)
    cam = Camera(
        src,
        digital_max=0xFF,
        analog_max=3.0,
        amplifier_bias=10,
        shot_noise=False,
        gaussian_noise_std=0.0,
        exposure=0.1 * u.ms,
    )

    # read the image from the camera, there should be an offset and small quantization error
    assert np.allclose(cam.read(), (255 / 3.0) * img + 10, atol=0.5)

    # change the exposure time and read again.
    # the image should be the same, but the exposure time should be different, and some pixels saturated
    cam.exposure = cam.exposure * 10
    assert np.allclose(cam.read(), np.minimum((255 / 3.0 * 10) * img + 10, 0xFF), atol=0.5)

    # perform HDR imaging with the camera, this should return the image without saturation
    # start with a trivial test, only one exposure, no background subtraction
    hdr = HDRCamera(cam, exposure_factors=(1.0,), background=0, saturation_threshold=255)
    assert np.allclose(hdr.read(), cam.read())

    hdr = HDRCamera(cam, exposure_factors=(1.0, 0.1, 0.01), background=10, saturation_threshold=250)
    frame = hdr.read()

    atol = (3 * 0.5) / 1.11 * 10  # worst case rounding error
    assert np.allclose(frame, (255 / 3.0 * 10) * img, atol=atol)
