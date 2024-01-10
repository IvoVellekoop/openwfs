import pytest
import time
import numpy.random
import pytest
from ..openwfs.slm import SLM, Patch, geometry
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u


class TestSLM:
    # just some values for testing different gray value levels
    GVAL1 = 173
    GVAL2 = 15
    GVAL3 = 23
    VAL1 = 2 * np.pi / 256 * GVAL1
    VAL2 = 2 * np.pi / 256 * GVAL2

    @pytest.fixture
    def windowed_slm(self):
        slm = SLM(shape=(100, 200), pos=(20, 10))
        mask = np.ones(slm.shape)
        mask[:, :50] = 0
        mask[:, 150:] = 0
        return slm, mask

    def test_windowed(self, windowed_slm):
        (slm, mask) = windowed_slm

        # create a new windowed-mode SLM and check if size and position match the specification
        assert slm.shape == (100, 200)
        assert slm.pos == (20, 10)

        # check if frame buffer has correct size
        fb_texture = slm._frame_patch._textures[Patch.PHASES_TEXTURE]
        assert fb_texture.shape == slm.shape

        # move window (we cannot really check this)
        slm.pos = (2, 1)
        assert slm.pos == (2, 1)

        # check default LUT
        assert slm.lookup_table[-1] == 255
        assert slm.lookup_table[0] == 0.0
        assert_allclose(slm.lookup_table, np.arange(0, 256))

        # put homogeneous phase on slm and read back
        slm.set_phases(TestSLM.VAL1)

        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL1 * mask)
        assert_allclose(slm.get_pixels('phase'), TestSLM.VAL1 * mask)
        slm.set_phases(TestSLM.VAL2, update=False)
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL1 * mask)  # nothing changed yet
        assert_allclose(slm.get_pixels('phase'), TestSLM.VAL1 * mask)
        slm.update()
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)
        assert_allclose(slm.get_pixels(), TestSLM.VAL2 * mask)

    @pytest.mark.skip(reason="This test is skipped by default because it causes the screen to flicker, which may "
                             "affect people with epilepsy.")
    def test_refresh_rate(self):
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
        assert_allclose(refresh_rate.to_value(u.Hz), actual_refresh_rate.to_value(u.Hz), rtol=1e-2)

    def test_get_pixels(self):
        width = 73
        height = 99
        slm = SLM(SLM.WINDOWED, shape=(height, width))
        slm.transform = geometry.fill_transform(slm, 'full')  # fill full screen exactly (anisotropic coordinates
        pattern = numpy.eye(*slm.shape) * TestSLM.VAL1
        pattern[-10:, -10:] = TestSLM.VAL1
        slm.set_phases(np.array(pattern, dtype='float32'))
        read_back = slm.get_pixels('gray_value')
        assert_allclose(pattern, read_back / 256 * 2 * np.pi)
        read_back = slm.get_pixels('phase')
        assert_allclose(pattern, read_back)

    def test_lookup_table(self, windowed_slm):
        (slm, mask) = windowed_slm

        # put homogeneous phase on slm and read back
        slm.set_phases(TestSLM.VAL2)
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)

        # change lookup table for one gray value
        lut = slm.lookup_table
        lut[TestSLM.GVAL2] = TestSLM.GVAL3
        slm.update()
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL2 * mask)
        slm.lookup_table = lut
        assert_allclose(slm.get_pixels('gray_value'), TestSLM.GVAL3 * mask)
