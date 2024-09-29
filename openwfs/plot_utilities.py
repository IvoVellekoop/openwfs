from astropy import units as u
from matplotlib import pyplot as plt

from .core import Detector
from .utilities import get_extent


def grab_and_show(cam: Detector, axis=None):
    return imshow(cam.read(), axis=axis)


def imshow(data, axis=None):
    extent = get_extent(data)
    e0 = scale_prefix(extent[0])
    e1 = scale_prefix(extent[1])
    if axis is None:
        plt.imshow(data, extent=(0.0, e1.value, 0.0, e0.value), cmap="gray")
        plt.colorbar()
        axis = plt.gca()
    else:
        axis.imshow(data, extent=(0.0, e1.value, 0.0, e0.value), cmap="gray")
    plt.ylabel(e0.unit.to_string())
    plt.xlabel(e1.unit.to_string())
    plt.show(block=False)
    plt.pause(0.1)
    return axis


def scale_prefix(value: u.Quantity) -> u.Quantity:
    """Scale a quantity to the most appropriate prefix unit."""
    if value.unit.physical_type == "length":
        if value < 100 * u.nm:
            return value.to(u.nm)
        if value < 100 * u.um:
            return value.to(u.um)
        if value < 100 * u.mm:
            return value.to(u.mm)
        else:
            return value.to(u.m)
    elif value.unit.physical_type == "time":
        if value < 100 * u.ns:
            return value.to(u.ns)
        if value < 100 * u.us:
            return value.to(u.us)
        if value < 100 * u.ms:
            return value.to(u.ms)
        else:
            return value.to(u.s)
    else:
        return value
