import time

import astropy.units as u
import cv2
import glfw
import numpy as np
import pytest

from openwfs.devices import is_loaded
from openwfs.devices.slm import SLM, Patch, geometry
from openwfs.utilities import Transform, project
from openwfs.simulation.slm import SLM as SimSLM
from numpy.testing import assert_allclose
from openwfs.utilities.utilities import get_pixel_size, set_extent
from openwfs.simulation.microscope import Microscope
from openwfs.simulation.mockdevices import StaticSource
from astropy.units import Quantity

if not is_loaded(glfw):
    pytest.skip(glfw.message, allow_module_level=True)

glfw.init()
if not glfw.get_monitors():
    pytest.skip("No monitors found", allow_module_level=True)


# just some values for testing different gray value levels
GVAL1 = 173
GVAL2 = 15
GVAL3 = 23
VAL1 = 2 * np.pi / 256 * GVAL1
VAL2 = 2 * np.pi / 256 * GVAL2


@pytest.fixture(params=[True, False])
def slm(request) -> SLM:
    hidden = request.param
    slm = SLM(monitor_id=0, shape=(100, 200), pos=(20, 10), coordinate_system="full", hidden=hidden)
    return slm


def test_create_windowed(slm):
    """Tests creating a windowed SLM. We should be able to make multiple windowed SLMs, but only one full screen SLM."""

    # check if size and position match the specification
    assert slm.shape == (100, 200)
    assert slm.position == (20, 10)
    assert slm.transform == Transform.identity()
    assert slm.coordinate_system == "full"

    # check if frame buffer has correct size
    fb_texture = slm._frame_buffer._textures[Patch._PHASES_TEXTURE]
    assert fb_texture.get_data().shape == slm.shape

    # check if the refresh rate got set
    assert int(slm.refresh_rate.to_value(u.Hz)) != glfw.DONT_CARE

    # move window (we cannot really check this, but we just check if no error occurs)
    slm.position = (2, 1)
    assert slm.position == (2, 1)

    # we should be able to construct a second windowed SLM
    _slm2 = SLM(0)

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
    _slm2 = None
    slm.monitor_id = 1
    full_screen_shape = slm.shape
    assert full_screen_shape[0] > 200  # assuming we don't have a tiny monitor
    assert full_screen_shape[1] > 200  # assuming we don't have a tiny monitor
    assert slm.refresh_rate.to_value(u.Hz) != glfw.DONT_CARE

    # we cannot make a windowed slm now because the full screen is already taken
    with pytest.raises(RuntimeError):
        slm3 = SLM(0)  # noqa: not used, just to check if an error occurs

    # make the slm windowed again
    # at the moment, the size of the window does not need to be conserved
    slm.monitor_id = 0
    assert slm.shape[0] < full_screen_shape[0]
    assert slm.shape[1] < full_screen_shape[1]

    # we can now make a second windowed slm
    slm4 = SLM(0)  # noqa: not used, just to check if an error occurs


# noinspection DuplicatedCode
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
    pixels = slm.pixels.read() / 64
    phases = slm.phases.read() * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:25, :100], 1)
    assert np.allclose(pixels[25:, :100], 0)
    assert np.allclose(pixels[:25, 100:], 2)
    assert np.allclose(pixels[25:, 100:], 3)

    # now change the transform to 'short' to fit the pattern to a centered square, with the height of the
    # SLM.
    # Then check if the pattern is displayed correctly
    slm.coordinate_system = "short"  # does not trigger an update
    assert np.all(slm.pixels.read() / 64 == pixels)
    slm.update()

    pixels = slm.pixels.read() / 64
    phases = slm.phases.read() * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:, :50], 0)
    assert np.allclose(pixels[:, 150:], 0)
    assert np.allclose(pixels[:25, 50:100], 1)
    assert np.allclose(pixels[25:, 50:100], 0)
    assert np.allclose(pixels[:25, 100:150], 2)
    assert np.allclose(pixels[25:, 100:150], 3)

    # now change the transform to 'long' to fit the pattern to a centered square, with the width of the
    # SLM, causing part of the texture to be mapped outside the window.
    slm.coordinate_system = "long"  # does not trigger an update
    assert np.all(slm.pixels.read() / 64 == pixels)
    slm.update()

    pixels = slm.pixels.read() / 64
    phases = slm.phases.read() * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    assert np.allclose(pixels[:, :100], 0)
    assert np.allclose(pixels[:, 100:], 3)

    # test zooming the pattern
    slm.coordinate_system = "short"
    slm.transform = Transform.zoom(0.8)
    slm.update()

    pixels = slm.pixels.read() / 64
    phases = slm.phases.read() * 4 / (2 * np.pi)
    assert np.allclose(pixels, phases)
    sub = pixels[10:90, 60:140].copy()
    pixels[10:90, 60:140] = 0
    assert np.allclose(pixels, 0)
    assert np.allclose(sub[:20, :40], 1)
    assert np.allclose(sub[20:, :40], 0)
    assert np.allclose(sub[:20, 40:], 2)
    assert np.allclose(sub[20:, 40:], 3)


@pytest.mark.skip(
    reason="This test is skipped by default because it causes the screen to flicker, which may "
    "affect people with epilepsy."
)
def test_refresh_rate():
    slm = SLM(1, latency=0, duration=0)
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
    slm.coordinate_system = "full"  # fill full screen exactly (anisotropic coordinates
    pattern = np.random.uniform(size=(height, width)) * 2 * np.pi
    slm.set_phases(pattern)
    read_back = slm.pixels.read()
    diff = pattern - read_back / 256 * 2 * np.pi
    diff[diff > np.pi] -= 2 * np.pi
    assert np.allclose(diff, 0, atol=2 * np.pi / 256)

    read_back = slm.phases.read()
    diff = pattern - read_back
    diff[diff > np.pi] -= 2 * np.pi
    assert np.allclose(diff, 0, atol=1e-6)


def test_lookup_table(slm):
    # verify that we cannot set properties that do not exist on the object
    with pytest.raises(AttributeError):
        slm.lut = np.arange(256)  # noqa - the field 'lut' does not exist

    # resize the SLM and put a linear ramp on it
    slm.shape = (1, 256)
    slm.set_phases(np.arange(256).reshape(1, 256) * 2 * np.pi / 256)

    # read back the pixels and verify conversion to gray values
    pixels = slm.pixels.read()
    assert np.allclose(pixels, np.arange(256))

    # set a lookup table that is a random permutation of the gray values
    lut = np.arange(256)
    np.random.shuffle(lut)
    slm.lookup_table = lut

    # nothing changes until we call update
    assert np.all(pixels == slm.pixels.read())

    slm.update()
    pixels = slm.pixels.read()
    assert np.allclose(pixels, lut)

    # test phase wrapping
    slm.set_phases(np.arange(256).reshape(1, 256) * 2 * np.pi / 256 + 2 * np.pi)
    pixels_wrapped = slm.pixels.read()
    assert np.allclose(pixels, pixels_wrapped)


def test_multi_patch(slm):
    slm.shape = (2, 2)
    slm.patches.append(Patch(slm))
    pattern1 = np.array(((5, 7), (111, 13)))
    pattern2 = np.array(((29, 31), (137, 300)))
    slm.patches[0].set_phases(pattern1 * 2 * np.pi / 256)
    slm.patches[1].set_phases(pattern2 * 2 * np.pi / 256)

    # with additive_blend, the patterns should be added and wrapped to 2 pi (256)
    slm.patches[1].additive_blend = True
    slm.update()
    assert np.allclose(slm.pixels.read(), (pattern1 + pattern2) % 256)

    # without additive_blend, the second patch overwrites the pixels of the first one
    slm.patches[1].additive_blend = False
    slm.update()
    assert np.allclose(slm.pixels.read(), pattern2 % 256)

    # disable the second patch and check if the first one is still there
    slm.patches[1].enabled = False
    slm.update()
    assert np.allclose(slm.pixels.read(), pattern1 % 256)


def test_circular_geometry(slm):
    slm.shape = (200, 200)
    slm.set_phases(np.linspace(0, 2 * np.pi, 70, endpoint=False).reshape(1, -1))
    radii = (0.0, 0.25, 0.50, 1.0)
    segments = (10, 20, 40)
    slm.patches[0].geometry = geometry.circular(radii=radii, segments_per_ring=segments)
    slm.update()

    # read back the pixels and verify conversion to gray values
    pixels = np.rint(slm.pixels.read() / 256 * 70)
    polar_pixels = cv2.warpPolar(pixels, (100, 40), (99.5, 99.5), 100, cv2.WARP_POLAR_LINEAR)

    assert np.allclose(
        polar_pixels[:, 3:24],
        np.repeat(np.flip(np.arange(0, 10)), 4).reshape((-1, 1)),
        atol=1,
    )
    assert np.allclose(
        polar_pixels[:, 27:47],
        np.repeat(np.flip(np.arange(10, 30)), 2).reshape((-1, 1)),
        atol=1,
    )
    assert np.allclose(
        polar_pixels[:, 53:97],
        np.repeat(np.flip(np.arange(30, 70)), 1).reshape((-1, 1)),
        atol=1,
    )


def test_calibrated_slm_initialization():
    """
    Test the initialization of the CalibratedSLM class.
    """
    physical_size = Quantity((2 * u.um, 2 * u.um))
    shape = (100, 100)

    slm = SLM(physical_size=physical_size, shape=shape, monitor_id=0)

    assert np.allclose(slm.physical_size, physical_size)
    assert slm.shape == shape
    assert slm.monitor_id == 0

    slm.physical_size = Quantity((3 * u.um, 3 * u.um))
    assert np.allclose(
        slm.physical_size,
        Quantity(
            (
                3 * u.um,
                3 * u.um,
            )
        ),
    ), "Physical size setter did not update the physical size correctly."


def test_cslm_field_amplitude():
    """
    Test the field amplitude of the CalibratedSLM class.
    """
    field_amplitude = np.ones((100, 100)) * 10
    field_amplitude[30:70, 30:70] = 0
    slm = SLM(
        physical_size=(2 * u.um, 2 * u.um),
        amplitude=field_amplitude,
        shape=(100, 100),
        monitor_id=0,
    )

    assert np.allclose(slm.amplitude, field_amplitude), "The field amplitude does not match the expected value."

    # should throw an error if the amplitude shape does not match the SLM shape
    try:
        field_amplitude = np.ones((99, 99))
        slm = SLM(
            physical_size=(2 * u.um, 2 * u.um),
            amplitude=field_amplitude,
            shape=(100, 100),
            monitor_id=0,
        )
    except ValueError as e:
        assert str(e) == "amplitude must have the same shape as the SLM shape."


def test_slm_mockSLM_equivalence():
    """
    Test the equivalence of the SLM and SimSLM classes.
    """
    slm = SimSLM(shape=(7, 14))
    slm2 = SLM(shape=(7, 14), monitor_id=0, coordinate_system="full")

    # use random seed and set phases
    rng = np.random.default_rng(seed=42)
    phases = rng.uniform(0, 2 * np.pi, size=(7, 14))

    slm.set_phases(phases)
    slm2.set_phases(phases)

    assert_allclose(slm.phases.read(), phases, rtol=1e-2, atol=1e-2)
    assert_allclose(slm2.phases.read(), phases, rtol=1e-1, atol=1e-2)
    assert_allclose(slm.phases.read(), slm2.phases.read(), rtol=1e-2, atol=1e-2)
    assert_allclose(slm.field.read(), slm2.field.read(), rtol=1e-2, atol=1e-2)
    assert_allclose(slm.pixels.read(), slm2.pixels.read(), rtol=1e-2, atol=1e-2)

    data = np.ones((7, 14))
    data[3, 3] = 0
    source = StaticSource(data=data, pixel_size=1 * u.um)
    mic = Microscope(
        source=source,
        numerical_aperture=0.8,
        wavelength=500 * u.nm,
        immersion_refractive_index=1.33,
        incident_field=slm.field,
    )
    mic2 = Microscope(
        source=source,
        numerical_aperture=0.8,
        wavelength=500 * u.nm,
        immersion_refractive_index=1.33,
        incident_field=slm2.field,
    )

    assert_allclose(mic2.read(), mic.read(), rtol=1e-2, atol=1e-2)
    assert_allclose(get_pixel_size(slm.phases.read()), get_pixel_size(slm2.phases.read()), rtol=1e-2, atol=1e-2)
    assert_allclose(get_pixel_size(slm.field.read()), get_pixel_size(slm2.field.read()), rtol=1e-2, atol=1e-2)
    assert_allclose(get_pixel_size(slm.pixels.read()), get_pixel_size(slm2.pixels.read()), rtol=1e-2, atol=1e-2)

    # test that physical size is handled correctly and that transform in mock microscope is applied correctly:
    slm2 = SLM(shape=(7, 14), monitor_id=0, coordinate_system="full", physical_size=(10, 5) * u.mm)

    slm2.set_phases(phases)

    assert_allclose(slm.phases.read(), phases, rtol=1e-2, atol=1e-2)
    assert_allclose(slm2.phases.read(), phases, rtol=1e-1, atol=1e-2)
    assert_allclose(slm.phases.read(), slm2.phases.read(), rtol=1e-2, atol=1e-2)
    assert_allclose(slm.field.read(), slm2.field.read(), rtol=1e-2, atol=1e-2)
    assert_allclose(slm.pixels.read(), slm2.pixels.read(), rtol=1e-2, atol=1e-2)

    data = np.ones((7, 14))
    data[3, 3] = 0
    source = StaticSource(data=data, pixel_size=1 * u.um)
    mic = Microscope(
        source=source,
        numerical_aperture=0.8,
        wavelength=500 * u.nm,
        immersion_refractive_index=1.33,
        incident_field=slm.field,
    )
    mic2 = Microscope(
        source=source,
        numerical_aperture=0.8,
        wavelength=500 * u.nm,
        immersion_refractive_index=1.33,
        incident_field=slm2.field,
    )

    assert_allclose(mic2.read(), mic.read(), rtol=1e-2, atol=1e-2)

    # the following is used in the mock microscope to convert the physical size of the SLM to normalized coordinates.
    # We test that this is done correctly.
    extent_physical = Quantity((10, 5), u.mm)
    x_physical = set_extent(phases, extent_physical)
    x_norm = set_extent(phases, 2)

    assert np.allclose(x_norm, x_physical, rtol=1e-2, atol=1e-2)
    x_norm_projected = project(
        x_norm,
        out_extent=2,
        out_shape=(7, 14),
        interp=cv2.INTER_LINEAR,
    )

    x_physical_projected = project(
        x_physical,
        out_extent=2,
        out_shape=(7, 14),
        transform=Transform(np.diag(2 / extent_physical)),
        interp=cv2.INTER_LINEAR,
    )

    assert np.allclose(x_norm_projected, x_physical_projected, rtol=1e-2, atol=1e-2)


def test_slm_update_amplitude():
    # test if amplitude of field is correctly updated when changing the amplitude property of the SLM

    # use random seed and set phases
    rng = np.random.default_rng(seed=42)
    amp = rng.uniform(size=(7, 14))
    slm = SLM(monitor_id=0, shape=(7, 14), physical_size=(70, 140) * u.mm, amplitude=amp)

    assert np.allclose(np.abs(slm.field.read()), amp)

    amp = rng.uniform(size=(7, 14))
    slm.amplitude = amp

    assert np.allclose(np.abs(slm.field.read()), amp)
