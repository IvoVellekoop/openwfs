import pytest

from openwfs.slm import SLM, Patch
import numpy as np
from numpy.testing import assert_allclose


class TestSLM:
    def test_windowed(self):
        # create a new windowed-mode SLM and check if size and position match the specification
        slm = SLM(width=200,height=100,left=20,top=10)
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
        GVAL1 = 173
        GVAL2 = 15
        GVAL3 = 23
        VAL1 = 2 * np.pi / 256 * GVAL1
        VAL2 = 2 * np.pi / 256 * GVAL2
        slm.phases = VAL1
        slm.update()
        mask = np.ones((slm.height, slm.width))
        mask[:, :50] = 0
        mask[:, 150:] = 0

        assert_allclose(slm.get_pixels(), GVAL1 * mask)
        #assert_allclose(slm.get_pixels('phase'), VAL1 * mask)
        slm.phases = VAL2
        assert_allclose(slm.get_pixels(), GVAL1 * mask)  # nothing changed yet
        #assert_allclose(slm.get_pixels('phase'), VAL1 * mask)
        slm.update()
        assert_allclose(slm.get_pixels(), GVAL2 * mask)
        #assert_allclose(slm.get_pixels('phase'), VAL2 * mask)

        # change lookup table for one gray value
        slm.lookup_table[GVAL2] = GVAL3 / 255.0
        assert_allclose(slm.get_pixels(), GVAL2 * mask)
        slm.update()
        assert_allclose(slm.get_pixels(), GVAL3 * mask)

    def test_monitor_id(self):
        assert False

    def test_width(self):
        assert False

    def test_height(self):
        assert False

    def test_left(self):
        assert False

    def test_left(self):
        assert False

    def test_top(self):
        assert False

    def test_top(self):
        assert False

    def test_refresh_rate(self):
        assert False

    def test__set_default_video_mode(self):
        assert False

    def test__set_actual_video_mode(self):
        assert False

    def test__create_window(self):
        assert False

    def test_update(self):
        assert False

    def test_idle_time(self):
        assert False

    def test_idle_time(self):
        assert False

    def test_settle_time(self):
        assert False

    def test_settle_time(self):
        assert False

    def test_wait(self):
        assert False

    def test_reserve(self):
        assert False

    def test_transform(self):
        assert False

    def test_transform(self):
        assert False

    def test_lookup_table(self):
        assert False

    def test_lookup_table(self):
        assert False

    def test_phases(self):
        assert False

    def test_phases(self):
        assert False
