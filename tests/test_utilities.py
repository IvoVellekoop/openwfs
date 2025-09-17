import astropy.units as u
import numpy as np

from ..openwfs.utilities import (
    set_pixel_size,
    get_pixel_size,
    place,
    Transform,
    project,
)


def test_to_matrix():
    # Create a transform object
    transform = Transform(
        transform=((1, 2), (3, 4)),
        source_origin=(0.0, 0.0) * u.m,
        destination_origin=(0.001, 0.002) * u.mm,
    )

    # Define the expected output matrix for same input and output pixel sizes
    expected_matrix = ((1, 2, 1), (3, 4, 1), (0, 0, 1))
    result_matrix = transform.to_matrix((1, 2) * u.um, (1, 2) * u.um)
    assert result_matrix.shape == (3, 3)
    assert np.allclose(result_matrix, expected_matrix)

    # Repeat for different input and output pixel sizes
    expected_matrix = ((0.5, 4, 1), (1.5, 8, 1), (0, 0, 1))
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


def test_find_shift():
    im = np.zeros((8, 8))
    im[2, 3] = 1  
    assert find_shift(im, im) == (0, 0) # no shift 

    im2 = np.zeros((8, 8))
    im2[3, 5] = 1  # shift by (1,2)
    assert find_shift(im, im2) == (1, 2)    

    im3 = np.zeros((8, 8))
    im3[1, 1] = 1  # shift by (-1,-2)
    assert find_shift(im, im3) == (-1, -2)  

    im1 = np.zeros((8, 8, 8))   
    im1[1, 1, 1] = 1
    im2 = np.zeros((8, 8, 8))
    im2[2, 3, 0] = 1  # shift by (1,2,-1)
    assert find_shift(im1, im2) == (1, 2, -1)

    im1 = np.zeros((1,7))
    im1[0, 0] = 1
    im2 = np.zeros((1,7))
    im2[0, 3] = 1  # shift by (0,3)
    assert find_shift(im1, im2) == (0, 3)

    im1 = np.zeros((1,7))
    im1[0, 0] = 1
    im2 = np.zeros((1,7))
    im2[0, 4] = 1  # shift by (0,4) but should be interpreted as (-0,-3) due to periodic boundaries
    assert find_shift(im1, im2) == (0, -3)


def test_cross_correlation_mean_corrected():
    im1 = np.zeros((8, 8))
    im1[0, 0] = 1    
    im1[1, 1] = -1   
    im2 = np.zeros((8, 8))
    im2[0, 0] = 2      
    im2[1, 1] = -1
    im2[7, 7] = -1
    assert np.allclose(cross_correlation_mean_corrected(im1, im1), im2) 

    im1 = np.zeros((8, 8))
    with pytest.raises(ValueError, match="After mean correction, one of the arrays is all zeros"):
        cross_correlation_mean_corrected(im1, im1)


def test_Transformation_Matrix_SLM_to_World_Coordinates():
    data = np.zeros((100, 100, 8))  # Example data
    data[0,0,0] = 1  # Example modification to data
    data[0,10,1] = 1  # Example modification to data
    data[0,0,2] = 1  # Example modification to data
    data[20,0,3] = 1  # Example modification to data
    data[0,0,4] = 1  # Example modification to data
    data[0,5,5] = 1  # Example modification to data
    data[0,0,6] = 1  # Example modification to data
    data[3,0,7] = 1  # Example modification to data
    M , G = Transformation_Matrix_SLM_to_World_Coordinates(data)

    M_expected = np.array([[10, 0], [0, 20]])
    G_expected = np.array([[5, 0], [0, 3]])

    assert np.allclose(M, M_expected)
    assert np.allclose(G, G_expected)