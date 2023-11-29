import numpy as np
from ..openwfs.core import get_pixel_size, set_pixel_size
from ..openwfs.utilities import place, Transform, project
import astropy.units as u


def test_place():
    ps1 = (0.5, 2) * u.um
    src = set_pixel_size(np.random.uniform(size=(7, 8)), ps1)

    # place image in output with same extent and same pixel size
    assert np.all(get_pixel_size(src) == ps1)
    dst_trivial = place(src.shape, ps1, src)
    assert np.allclose(src, dst_trivial)
    assert np.all(get_pixel_size(dst_trivial) == ps1)

    # place image in output with larger extent and same pixel size.
    # the source will be padded with zeros (exactly one row and two columns on all sides)
    dst_pad = place((9, 12), ps1, src)
    assert np.all(dst_pad[0, :] == 0.0)
    assert np.all(dst_pad[-1, :] == 0.0)
    assert np.all(dst_pad[:, -2:] == 0.0)
    assert np.all(dst_pad[:, 0:2] == 0.0)

    # place the image back in an array of the original shape.
    # this should remove the padding
    dst_crop = place((7, 8), ps1, dst_pad)
    assert np.allclose(dst_crop, src)

    # use offset to shift the image
    dst_shift = place((7, 8), ps1, src, offset=ps1 * (2, 1))
    assert np.allclose(dst_shift[0:2, :], 0.0)
    assert np.allclose(dst_shift[:, 0], 0.0)
    assert np.allclose(dst_shift[2:, 1:], src[:-2, :-1])


def test_transform():
    ps1 = (0.5, 2) * u.um
    src = set_pixel_size(np.random.uniform(size=(7, 8)), ps1)

    id = Transform.identity()
    matrix = id.cv2_matrix(src.shape, ps1, src.shape, ps1)
    assert np.allclose(matrix, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    # shift both origins by same distance
    t0 = Transform(source_origin=-ps1 * (1.7, 2.2), destination_origin=-ps1 * (1.7, 2.2))
    dst0 = project(src.shape, ps1, src, t0)
    assert np.allclose(dst0, src)

    # shift source by (1,2) pixel
    t1 = Transform(source_origin=-ps1 * (1, 2))
    dst1a = place(src.shape, ps1, src, offset=ps1 * (1, 2))
    dst1b = project(src.shape, ps1, src, t1)
    assert np.allclose(dst1a, dst1b)
