import pytest
import numpy as np
from openwfs.utilities.patterns import (
    tilt,
    gaussian,
    disk,
    propagation,
    parabola,
    binary_grating,
    coordinate_range,
    zernike,
    noll_to_nm,
    nm_to_noll,
)
from openwfs.utilities import unitless
import astropy.units as u


@pytest.mark.parametrize("shape", [10, (7, 10)])
def test_tilt(shape):
    t = tilt(shape, extent=(2, 2), g=(5, 0.71))
    if np.size(shape) == 1:
        shape = (shape, shape)

    # for a default extent of 2.0, 2.0, the tilt should span 4 * t
    phase_diff0 = 5 * 4 * (1.0 - 1.0 / shape[0])
    phase_diff1 = 0.71 * 4 * (1.0 - 1.0 / shape[1])
    assert np.allclose(t[-1, 0] - t[0, 0], phase_diff0)
    assert np.allclose(t[-1, -1] - t[0, -1], phase_diff0)
    assert np.allclose(t[0, -1] - t[0, 0], phase_diff1)
    assert np.allclose(t[-1, -1] - t[-1, 0], phase_diff1)


def test_gaussian_disk_offset():
    shape = (101, 66)
    offset = (0.2, 0.3)
    waist = 0.25
    g = gaussian(shape, extent=(2, 2), waist=waist, offset=offset)
    argmax = np.unravel_index(np.argmax(g), g.shape)

    expected = np.round((np.array(offset) + 1) * (np.array(shape) - 1) / 2.0)
    assert np.allclose(expected, argmax)

    shape = (101, 101)
    d = disk(shape, radius=2 / 101, extent=(2, 2), offset=offset)
    arg_center = np.argwhere(d > 0.5)[0]
    expected = (np.array(offset) + 1) * (np.array(shape) - 1) / 2.0
    assert np.allclose(expected, arg_center)


def test_parabola():
    phi = parabola((11, 11), extent=(2, 2), alpha=0.5)
    assert np.allclose(phi[5, 5], 0)
    r = -1 + 1 / 11
    assert np.allclose(phi[0, 5], 0.5 * (r**2))
    assert np.allclose(phi[0, 0], 0.5 * (r**2 + r**2))


@pytest.mark.parametrize("extent", [2, 4])
def test_binary_grating(extent):
    extent = 2.0
    period = 0.1
    values = (1, 2)
    shape = (1000, 1)
    phi = binary_grating(shape, period, values, extent=extent, angle=0)

    # Check that the values are correct within the expected period
    first_up = np.argmax(phi)
    phi_2 = phi[first_up:]
    first_down = np.argmin(phi_2)
    np.allclose(phi_2[0], values[1])
    np.allclose(phi_2[first_down], values[0])
    p = first_down
    np.allclose(phi_2[p : p + first_down], values[0])
    np.allclose(phi_2[p + first_down : p + 2 * first_down], values[1])
    np.allclose(period, 2 * p / shape[0] * extent)

    phi_cte = binary_grating(shape, period, values, extent=extent, angle=0, round_period=True)
    # Check that round_period does not change the pattern when the period is already an integer number of pixels
    assert np.allclose(phi_cte, phi)

    phi_cte_2 = binary_grating(shape, 0.105, values, extent=extent, angle=0, round_period=True)

    # Ensure that phi_cte_2 is not the same as phi_cte (This only serves to check that the rounding is actually doing something, since the period is different, the pattern should be different)
    assert not np.allclose(phi_cte, phi_cte_2)
    tmp = (phi_cte_2[6:-6]).reshape(-1, 26)
    # Check that the pattern has the same number of pixels up and down with a constant period
    assert np.all(np.isclose(tmp[0:-1:2], tmp[0, :]))
    assert np.all(np.isclose(tmp[1:-1:2], tmp[1, :]))

    # Tests that rounds_period works fine with single valued shape and extent
    assert np.allclose(
        binary_grating(100, 0.1, values, extent=2, round_period=True),
        binary_grating((100, 100), 0.1, values, extent=(2, 2), round_period=True),
    )


@pytest.mark.parametrize("extent, refractive_index", [(2, 1.0), (1, 1.5)])
def test_propagation(extent, refractive_index):
    shape = 101
    d = 10 * u.um
    wavelength = 0.5 * u.um
    na = 1
    phi = propagation(shape, d, wavelength, na, refractive_index=refractive_index, extent=extent)

    r_r = coordinate_range(shape, extent)[0][0, 0]
    kr = 2 * np.pi / wavelength * na * r_r
    kz = np.sqrt((2 * np.pi * refractive_index / wavelength) ** 2 - kr**2)
    ref_value = unitless(d * kz)

    assert np.allclose(ref_value, phi[0, 50])


def test_propagation_vector_distance():
    # test that propagation can handle a vector of distances and returns the correct shape and values
    distances = np.linspace(-50, 20, 10) * u.um

    vector_result = propagation(
        shape=(100, 100),
        distance=distances,
        wavelength=532 * u.nm,
        numerical_aperture=0.8,
    )

    individual_results = []

    for distance in distances:
        result = propagation(
            shape=(100, 100),
            distance=distance,
            wavelength=532 * u.nm,
            numerical_aperture=0.8,
        )

        individual_results.append(result)

    loop_result = np.stack(individual_results, axis=-1)

    assert vector_result.shape == (100, 100, 10)
    np.testing.assert_allclose(vector_result, loop_result)


SHAPE = (512, 512)
EXTENT = (2.0, 2.0)


@pytest.fixture(scope="module")
def pupil():
    """Boolean mask for the sampled unit pupil."""
    y, x = coordinate_range(SHAPE, EXTENT)
    return unitless(x**2 + y**2) <= 1


@pytest.mark.parametrize("j", range(1, 11))
def test_zernike_has_unit_rms(j, pupil):
    """Every orthonormal Zernike mode must have unit RMS."""
    mode = zernike(SHAPE, j, EXTENT)

    rms = np.sqrt(np.mean(mode[pupil] ** 2))

    assert rms == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize(
    "j1, j2",
    [
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
        (9, 10),
    ],
)
def test_different_zernike_modes_are_orthogonal(j1, j2, pupil):
    """The pupil-averaged inner product of different modes must be zero."""
    mode1 = zernike(SHAPE, j1, EXTENT)
    mode2 = zernike(SHAPE, j2, EXTENT)

    inner_product = np.mean(mode1[pupil] * mode2[pupil])

    assert inner_product == pytest.approx(0.0, abs=5e-3)


def test_first_ten_zernikes_are_orthonormal(pupil):
    """Test the complete discrete Gram matrix of the first ten modes."""
    modes = np.array([zernike(SHAPE, j, EXTENT)[pupil] for j in range(1, 11)])

    # G[i, j] is the pupil-averaged inner product of modes i and j.
    gram_matrix = modes @ modes.T / pupil.sum()

    np.testing.assert_allclose(
        gram_matrix,
        np.eye(10),
        atol=5e-3,
    )


def test_primary_spherical_aberration():
    """
    Noll mode j=11 must equal the normalized spherical polynomial

        Z_11 = sqrt(5) * (6 rho^4 - 6 rho^2 + 1).
    """
    y, x = coordinate_range(SHAPE, EXTENT)
    rho = unitless(np.hypot(x, y))
    pupil = rho <= 1

    expected = np.where(
        pupil,
        np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1),
        0.0,
    )

    actual = zernike(SHAPE, j=11, extent=EXTENT)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zernike_is_zero_outside_pupil(pupil):
    """The generated pattern must be zero outside the unit pupil."""
    mode = zernike(SHAPE, j=11, extent=EXTENT)

    assert np.all(mode[~pupil] == 0.0)


@pytest.mark.parametrize(
    "j, nm",
    [(1, (0, 0)), (2, (1, 1)), (3, (1, 1)), (4, (2, 0)), (5, (2, 2)), (6, (2, 2)), (11, (4, 0))],
)
def test_noll_2_nm(j, nm):
    assert noll_to_nm(j) == nm


@pytest.mark.parametrize(
    "n, m, js",
    [(0, 0, (1,)), (1, 1, (2, 3)), (2, 0, (4,)), (2, 2, (5, 6)), (4, 0, (11,))],
)
def test_nm_to_noll(n, m, js):
    assert nm_to_noll(n, m) == js


@pytest.mark.parametrize("j", range(1, 50))
def test_noll_round_trip(j):
    assert j in nm_to_noll(*noll_to_nm(j))
    
