import time

import numpy.random
import pytest
from openwfs.slm import SLM, Patch, geometry
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Quantity



class TestSLM:
    # just some values for testing different gray value levels
    GVAL1 = 173
    GVAL2 = 15
    GVAL3 = 23
    VAL1 = 2 * np.pi / 256 * GVAL1
    VAL2 = 2 * np.pi / 256 * GVAL2

    @pytest.fixture
    def windowed_slm(self):
        slm = SLM(width=200,height=100,left=20,top=10)
        mask = np.ones((slm.height, slm.width))
        mask[:, :50] = 0
        mask[:, 150:] = 0
        return (slm, mask)

    def test_windowed(self, windowed_slm):
        (slm, mask) = windowed_slm

        # create a new windowed-mode SLM and check if size and position match the specification
        assert slm.width == 200
        assert slm.height == 100
        assert slm.left == 20
        assert slm.top == 10

        # check if frame buffer has correct size
        fb_texture = slm._frame_patch._textures[Patch.PHASES_TEXTURE]
        assert fb_texture.width == slm.width
        assert fb_texture.height == slm.height

        # move window
        slm.left = 1
        slm.top = 2
        assert slm.left == 1
        assert slm.top == 2

        # check default LUT
        assert slm.lookup_table[-1] == 1.0
        assert slm.lookup_table[0] == 0.0
        assert_allclose(slm.lookup_table, np.arange(0, 256)/255.0, )

        # put homogeneous phase on slm and read back
        slm.phases = TestSLM.VAL1
        slm.update()

        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL1 * mask)
        assert_allclose(slm.get_pixels('phase'), TestSLM.VAL1 * mask)
        slm.phases = TestSLM.VAL2
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL1 * mask)  # nothing changed yet
        assert_allclose(slm.get_pixels('phase'), TestSLM.VAL1 * mask)
        slm.update()
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)
        assert_allclose(slm.get_pixels(), TestSLM.VAL2 * mask)

    def test_monitor_id(self):
        assert False

    def test_refresh_rate(self):
        slm = SLM(1)
        slm.idle_time = 0
        slm.settle_time = 0
        refresh_rate = slm.refresh_rate
        slm.update()
        start = time.time_ns() * u.ns
        frame_count = 100
        for i in range(frame_count):
            slm.phases = 0.0 if (i % 2) else np.pi
            slm.update()
        stop = time.time_ns() * u.ns
        del slm
        actual_refresh_rate = frame_count/(stop - start)
        assert_allclose(refresh_rate.to_value(u.Hz), actual_refresh_rate.to_value(u.Hz), rtol=1e-2)

    def test_get_pixels(self):
        width = 73
        height = 99
        slm = SLM(SLM.WINDOWED, width, height)
        slm.transform = geometry.fill_transform(slm, 'full') # fill full screen exactly (anisotropic coordinates
        pattern = numpy.eye(height, width) * TestSLM.VAL1
        pattern[-10:,-10:] = TestSLM.VAL1
        slm.phases = np.array(pattern, dtype='float32')
        slm.update()
        read_back = slm.get_pixels('gray_value')
        assert_allclose(pattern, read_back / 256 * 2 * np.pi)
        read_back = slm.get_pixels('phase')
        assert_allclose(pattern, read_back)

    def test_idle_time(self):
        assert False

    def test_settle_time(self):
        assert False

    def test_wait(self):
        assert False

    def test_reserve(self):
        assert False

    def test_transform(self):
        assert False

    def test_lookup_table(self, windowed_slm):
        (slm, mask) = windowed_slm

        # put homogeneous phase on slm and read back
        slm.phases = TestSLM.VAL2
        slm.update()
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)

        # change lookup table for one gray value
        slm.lookup_table[TestSLM.GVAL2] = TestSLM.GVAL3 / 255.0
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)
        slm.update()
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL3 * mask)

    def test_phases(self):
        assert False
