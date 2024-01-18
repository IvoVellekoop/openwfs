import time

import glfw
import numpy.random
import pytest
from ..openwfs.slm import SLM, Patch
from ..openwfs.utilities import Transform
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt  # for debugging
import numpy as np  # for debugging

# just some values for testing different gray value levels
GVAL1 = 173
GVAL2 = 15
GVAL3 = 23
VAL1 = 2 * np.pi / 256 * GVAL1
VAL2 = 2 * np.pi / 256 * GVAL2


@pytest.fixture
def slm() -> SLM:
    slm = SLM(monitor_id=0, shape=(100, 200), pos=(20, 10), transform='full')
    return slm


def test_create_windowed(slm):
    """Tests creating a windowed SLM. We should be able to make multiple windowed SLMs, but only one full screen SLM."""

    # check if size and position match the specification
    assert slm.shape == (100, 200)
    assert slm.pos == (20, 10)

    # check if frame buffer has correct size
    fb_texture = slm._frame_buffer._textures[Patch._PHASES_TEXTURE]
    assert fb_texture.shape == slm.shape

    # check if the refresh rate got set
    assert int(slm.refresh_rate.to_value(u.Hz)) != glfw.DONT_CARE

    # move window (we cannot really check this, but we just check if no error occurs)
    slm.pos = (2, 1)
    assert slm.pos == (2, 1)

    # we should be able to construct a second windowed SLM
    slm2 = SLM(0)

    # but not a full screen SLM on this window
    with pytest.raises(RuntimeError):
        SLM(1)

    # and we also expect an error when the window does not exist
    with pytest.raises(IndexError):
        SLM(999)

    # we also cannot make the slm full screen, because slm2 is still active
    with pytest.raises(RuntimeError):
        slm.monitor_id = 1

    # but if we delete slm2 first, we can make the slm full screen
    slm2 = None
    slm.monitor_id = 1
    full_screen_shape = slm.shape
    assert full_screen_shape[0] > 200  # assuming we don't have a tiny monitor
    assert full_screen_shape[1] > 200  # assuming we don't have a tiny monitor
    assert slm.refresh_rate.to_value(u.Hz) != glfw.DONT_CARE

    # we cannot make a windowed slm now because the full screen is already taken
    with pytest.raises(RuntimeError):
        slm3 = SLM(0)

    # make the slm windowed again
    # at the moment, the size of the window does not need to be conserved
    slm.monitor_id = 0
    assert slm.shape[0] < full_screen_shape[0]
    assert slm.shape[1] < full_screen_shape[1]

    # we can now make a second windowed slm
    slm4 = SLM(0)


def test_transform(slm):
    """Tests the transform property of the SLM class."""

    # construct a test pattern:
    # 1 | 1 | 2 | 2
    # --+---+---+--
    # 0 | 0 | 3 | 3
    # --+---+---+--
    # 0 | 0 | 3 | 3
    # --+---+---+--
    # 0 | 0 | 3 | 3
    pattern = np.array((1, 1, 2, 2, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3)).reshape((4, 4))
    assert pattern[1, 3] == 3

    # display the pattern on the SLM, using the 'full' transform
    slm.set_phases(2.0 * np.pi * pattern / 4.0)

    # check if the pattern is displayed correctly
    pixels = slm.get_pixels('gray_value') / 64
    phases = slm.get_pixels('phase') * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:25, :100], 1)
    assert np.allclose(pixels[25:, :100], 0)
    assert np.allclose(pixels[:25, 100:], 2)
    assert np.allclose(pixels[25:, 100:], 3)

    # now change the transform to 'short' to fit the pattern to a centered square, with the height of the
    # SLM.
    # Then check if the pattern is displayed correctly
    slm.transform = 'short'  # does not trigger an update
    assert np.alltrue(slm.get_pixels('gray_value') / 64 == pixels)
    slm.update()

    pixels = slm.get_pixels('gray_value') / 64
    phases = slm.get_pixels('phase') * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:, :50], 0)
    assert np.allclose(pixels[:, 150:], 0)
    assert np.allclose(pixels[:25, 50:100], 1)
    assert np.allclose(pixels[25:, 50:100], 0)
    assert np.allclose(pixels[:25, 100:150], 2)
    assert np.allclose(pixels[25:, 100:150], 3)

    # now change the transform to 'long' to fit the pattern to a centered square, with the width of the
    # SLM, causing part of the texture to be mapped outside of the window.
    slm.transform = 'long'  # does not trigger an update
    assert np.alltrue(slm.get_pixels('gray_value') / 64 == pixels)
    slm.update()

    pixels = slm.get_pixels('gray_value') / 64
    phases = slm.get_pixels('phase') * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:, :100], 0)
    assert np.allclose(pixels[:, 100:], 3)

    # test zooming the pattern
    slm.transform = 'short'
    slm.transform = Transform.zoom(0.8) @ slm.transform
    slm.update()

    pixels = slm.get_pixels('gray_value') / 64
    phases = slm.get_pixels('phase') * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    sub = pixels[10:90, 60:140].copy()
    pixels[10:90, 60:140] = 0
    assert np.allclose(pixels, 0)
    assert np.allclose(sub[:20, :40], 1)
    assert np.allclose(sub[20:, :40], 0)
    assert np.allclose(sub[:20, 40:], 2)
    assert np.allclose(sub[20:, 40:], 3)


# @pytest.mark.skip(reason="This test is skipped by default because it causes the screen to flicker, which may "
#                        "affect people with epilepsy.")
def test_refresh_rate():
    slm = SLM(1)
    slm.latency = 0
    slm.duration = 0
    refresh_rate = slm.refresh_rate
    slm.update()
    frame_count = 100

    # warm up
    for i in range(frame_count):
        slm.set_phases(0.0 if (i % 2) else np.pi)

    start = time.time_ns() * u.ns
    for i in range(frame_count):
        slm.set_phases(0.0 if (i % 2) else np.pi)
    stop = time.time_ns() * u.ns
    del slm
    actual_refresh_rate = frame_count / (stop - start)
    assert np.allclose(refresh_rate.to_value(u.Hz), actual_refresh_rate.to_value(u.Hz), rtol=1e-2)


def test_get_pixels():
    width = 73
    height = 99
    slm = SLM(SLM.WINDOWED, shape=(height, width))
    slm.transform = 'full'  # fill full screen exactly (anisotropic coordinates
    pattern = np.random.uniform(size=(height, width)) * 2 * np.pi
    slm.set_phases(pattern)
    read_back = slm.get_pixels('gray_value')
    diff = pattern - read_back / 256 * 2 * np.pi
    diff[diff > np.pi] -= 2 * np.pi
    assert np.allclose(diff, 0, atol=2 * np.pi / 256)

    read_back = slm.get_pixels('phase')
    diff = pattern - read_back
    diff[diff > np.pi] -= 2 * np.pi
    assert np.allclose(diff, 0, atol=1e-6)


def test_lookup_table(slm):
    # verify that we cannot set properties that do not exist on the object
    with pytest.raises(AttributeError):
        slm.lookup_table = np.arange(256)

    # resize the SLM and put a linear ramp on it
    slm.shape = (1, 256)
    slm.set_phases(np.arange(256).reshape(1, 256) * 2 * np.pi / 256)

    # read back the pixels and verify conversion to gray values
    pixels = slm.get_pixels('gray_value')
    assert np.allclose(pixels, np.arange(256))

    # set a lookup table that is a random permutation of the gray values
    lut = np.arange(256)
    np.random.shuffle(lut)
    slm.lut = lut

    # nothing changes until we call update
    assert np.alltrue(pixels == slm.get_pixels('gray_value'))

    slm.update()
    pixels = slm.get_pixels('gray_value')
    assert np.allclose(pixels, lut)
