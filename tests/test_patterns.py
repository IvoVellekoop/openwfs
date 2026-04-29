import pytest
import numpy as np
from openwfs.utilities.patterns import tilt, gaussian, disk, propagation, parabola, binary_grating, coordinate_range
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
    period = 0.1
    values = (1, 2)
    shape = (1000, 1)
    phi = binary_grating(shape, period, values, extent=extent, angle=0)
    first_up = np.argmax(phi)
    phi_2 = phi[first_up:]
    first_down = np.argmin(phi_2)
    np.allclose(phi_2[0], values[1])
    np.allclose(phi_2[first_down], values[0])
    p = first_down
    np.allclose(phi_2[p : p + first_down], values[0])
    np.allclose(phi_2[p + first_down : p + 2 * first_down], values[1])
    np.allclose(period, 2 * p / shape[0] * extent)


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
