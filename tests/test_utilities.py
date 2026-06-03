import astropy.units as u
import numpy as np
import pytest

from openwfs.utilities import (
    set_pixel_size,
    get_pixel_size,
    place,
    Transform,
    project,
    set_extent,
)

from openwfs.utilities.patterns import parabola

def test_to_matrix():
    # equal pixel sizes and identity matrix should result in identity matrix
    transform = Transform(
        transform=((1, 0), (0, 1)),
        source_origin=(0.0, 0.0) * u.m,
        destination_origin=(0, 0) * u.mm,
    )
    # Define the expected output matrix for same input and output pixel sizes
    expected_matrix = np.eye(3)
    result_matrix = transform.to_matrix((1, 2) * u.um, (1, 2) * u.um)

    assert result_matrix.shape == (3, 3)
    assert np.allclose(result_matrix, expected_matrix)

    # Non equal pixel sizes and identity matrix should be scaled by pixel ratios
    transform = Transform(
        transform=((1, 0), (0, 1)),
        source_origin=(0.0, 0.0) * u.m,
        destination_origin=(0, 0) * u.mm,
    )
    # Define the expected output matrix for same input and output pixel sizes
    # if you want to keep NA coordinates at the same location (identity transform above), but pixels are halv
    # as small in x direction, you need to move scale by 2 in the x direction, so that the same physical distance
    # corresponds to 2 pixels instead of 1 pixel.
    expected_matrix = np.eye(3)
    expected_matrix[0, 0] = 2 * u.um / (1 * u.um)
    result_matrix = transform.to_matrix((2, 2) * u.um, (1, 2) * u.um)

    assert result_matrix.shape == (3, 3)
    assert np.allclose(result_matrix, expected_matrix)

    # test off diagonal matrix with anisotropic pixel sizes.
    # Create a transform object
    transform = Transform(
        transform=((1, 2), (3, 4)),
        source_origin=(0.0, 0.0) * u.m,
        destination_origin=(0, 0) * u.mm,
    )
    # Coordinate (1,0) in physical coordinates is transformed to (1,3) in physical coordinates
    #
    # (1 um ,0) correponds to (1 , 0 ) in pixel coordinates, and (1um, 3um) corresponds to (1, 1.5) in pixel coordinates,
    # so the first column of the matrix should be (1, 1.5, 0)
    # Similarly (0,1 um) (corresponds to (0 , 0.5) in pixel coordinates) is transformed to (2um ,4 um)
    # (corresponds to (2, 2) in pixel coordinates, for pixel size (1um, 2um))), so the second column of the matrix should be (4, 4, 0)
    # Define the expected output matrix for same input and output pixel sizes
    expected_matrix = ((1, 4, 0), (1.5, 4, 0), (0, 0, 1))
    result_matrix = transform.to_matrix((1, 2) * u.um, (1, 2) * u.um)
    assert result_matrix.shape == (3, 3)
    assert np.allclose(result_matrix, expected_matrix)

    # now add a translation of 1 , 1 pixel, for pixel sizes (1um, 2um), this corresponds to a translation of (1um, 2um) in physical coordinates, so the last column of the matrix should be (1, 1.5, 1)
    # Create a transform object
    transform = Transform(
        transform=((1, 2), (3, 4)),
        source_origin=(0.0, 0.0) * u.m,
        destination_origin=(0.001, 0.002) * u.mm,
    )

    # Define the expected output matrix for same input and output pixel sizes
    expected_matrix = ((1, 4, 1), (1.5, 4, 1), (0, 0, 1))
    result_matrix = transform.to_matrix((1, 2) * u.um, (1, 2) * u.um)
    assert result_matrix.shape == (3, 3)
    assert np.allclose(result_matrix, expected_matrix)

    # Repeat for different input and output pixel sizes
    expected_matrix = ((0.5, 8, 1), (0.75, 8, 1), (0, 0, 1))
    result_matrix = transform.to_matrix((0.5, 4) * u.um, (1, 2) * u.um)

    assert np.allclose(result_matrix, expected_matrix)

    # Test center correction. The center of the source image should be mapped to the center of the destination image
    src = (17, 18)
    dst = (13, 14)
    src_center = np.array((0.5 * (src[1] - 1), 0.5 * (src[0] - 1), 1.0))
    dst_center = np.array((0.5 * (dst[1] - 1), 0.5 * (dst[0] - 1)))
    transform = Transform()
    result_matrix = transform.cv2_matrix(
        source_shape=src,
        source_pixel_size=(1, 1),
        destination_shape=dst,
        destination_pixel_size=(1, 1),
    )
    assert np.allclose(result_matrix @ src_center, dst_center)

    # Test center correction. The center of the source image should be mapped to the center of the destination image
    transform = Transform()  # transform=((1, 2), (3, 4)))
    result_matrix = transform.cv2_matrix(
        source_shape=src,
        source_pixel_size=(0.5, 4) * u.um,
        destination_shape=dst,
        destination_pixel_size=(1, 2) * u.um,
    )
    assert np.allclose(result_matrix @ src_center, dst_center)

    # Also check openGL matrix (has y-axis flipped and extra row and column)
    expected_matrix = ((1, 2, 1), (3, 4, 2), (0, 0, 1))
    transform = Transform(transform=((1, 2), (3, 4)), source_origin=(0, 0), destination_origin=(1, 2))
    result_matrix = transform.to_matrix((1, 1), (1, 1))
    assert np.allclose(result_matrix, expected_matrix)

    result_matrix = transform.opencl_matrix()
    expected_matrix = ((4, -2, 0, 0), (3, -1, 0, 0), (2, -1, 1, 0))
    assert np.allclose(result_matrix, expected_matrix)


def test_place():
    ps1 = (0.5, 2) * u.um
    src = set_pixel_size(np.random.uniform(size=(7, 8)), ps1)

    # place image in output with same extent and same pixel size
    assert np.all(get_pixel_size(src) == ps1)
    dst_trivial = place(src.shape, ps1, src)
    assert np.allclose(src, dst_trivial)
    assert np.all(get_pixel_size(dst_trivial) == ps1)

    ps1 = (0.5, 2) * u.um
    ps2 = (3.0, 1.0) * u.um
    src = set_pixel_size(np.zeros((7, 8)), ps1)
    with pytest.raises(NotImplementedError):
        set_pixel_size(src, ps2)

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

    # place image in output with same extent and different pixel size
    ps2 = (0.25, 1.0) * u.um
    dst_resample = place((14, 16), ps2, src)
    src_resample = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)
    assert np.allclose(dst_resample, src_resample)


def test_transform():
    ps1 = (0.5, 2) * u.um
    src = set_pixel_size(np.random.uniform(size=(7, 8)), ps1)

    identity = Transform()
    matrix = identity.cv2_matrix(src.shape, ps1, src.shape, ps1)
    assert np.allclose(matrix, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    # shift both origins by same distance
    t0 = Transform(source_origin=-ps1 * (1.7, 2.2), destination_origin=-ps1 * (1.7, 2.2))
    dst0 = project(
        src,
        source_extent=ps1 * np.array(src.shape),
        transform=t0,
        out_extent=ps1 * np.array(src.shape),
        out_shape=src.shape,
    )
    assert np.allclose(dst0, src)

    # shift source by (1,2) pixel
    t1 = Transform(source_origin=-ps1 * (1, 2))
    dst1a = place(src.shape, ps1, src, offset=ps1 * (1, 2))
    dst1b = project(
        src,
        source_extent=ps1 * np.array(src.shape),
        transform=t1,
        out_extent=ps1 * np.array(src.shape),
        out_shape=src.shape,
    )
    assert np.allclose(dst1a, dst1b)

    # test that when flipping and rotating, the centre stays the centre, not matter pixel shape
    transform = Transform(transform = np.array([[0, -1], [-2, 0]]))
    extent = (5,9)
    pat = -parabola(extent, 0.5, extent)
    centre = (pat.shape[0] // 2, pat.shape[1] // 2)
    max_loc = np.unravel_index(np.argmax(pat), pat.shape)
    assert max_loc == centre, f"Expected maximum pixel location to be at the center {centre}, but got {max_loc}"


    pat = set_extent(pat, extent)
    pat_extent = (9, 9)
    pat_big = project(source = pat, out_extent=pat_extent, out_shape=(13, 5), transform=transform)
    max_loc_big = np.unravel_index(np.argmax(pat_big), pat_big.shape)
    centre_big = (pat_big.shape[0] // 2, pat_big.shape[1] // 2)
    assert max_loc_big == centre_big, f"Expected maximum pixel location to be at the center {centre_big}, but got {max_loc_big}"


def test_zoom():
    transform = Transform.zoom(2.0)
    vector = (0.3, 0.4)
    result = transform.apply(vector)
    expected_result = (0.6, 0.8)
    assert np.allclose(result, expected_result)


def test_inverse():
    transform = Transform(
        transform=((0.1, 0.2), (-0.25, 0.33)),
        source_origin=(0.12, 0.15),
        destination_origin=(0.23, 0.33),
    )
    vector = (0.3, 0.4)
    result = transform.apply(vector)

    inverse = transform.inverse()
    transformed_back = inverse.apply(result)
    assert np.allclose(vector, transformed_back)
    assert np.allclose(inverse @ transform @ vector, vector)
    assert np.allclose(transform @ inverse @ vector, vector)


def test_utilities_microscope():
    import openwfs.utilities as owf_u
    import openwfs.simulation as owf_s

    mic, slm, src = owf_u.tests.get_test_microscope()
    assert type(mic) == owf_s.Microscope
    assert type(slm) == owf_s.SLM
    assert type(src) == owf_s.StaticSource
