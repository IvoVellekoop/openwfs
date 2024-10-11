from typing import Tuple, Union, Optional, Dict

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from numpy import ndarray as nd

from .core import Detector
from .utilities import get_extent


# TODO: needs review and documentation. Remove single-use functions, simplify code.


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


def slope_step(a: nd, width: Union[nd, float]) -> nd:
    """
    A sloped step function from 0 to 1.

    Args:
        a: Input array
        width: width of the sloped step.

    Returns:
        An array the size of a, with the result of the sloped step function.
    """
    return (a >= width) + a / width * (0 < a) * (a < width)


def linear_blend(a: nd, b: nd, blend: Union[nd, float]) -> nd:
    """
    Return a linear, element-wise blend between two arrays a and b.

    Args:
        a: Input array a.
        b: Input array b.
        blend: Blend factor. Value of 1.0 -> return a. Value of 0.0 -> return b.

    Returns:
        A linear combination of a and b, corresponding to the blend factor. a*blend + b*(1-blend)
    """
    return a * blend + b * (1 - blend)


def complex_to_rgb(array: nd, scale: Optional[Union[float, nd]] = None, axis: int = 2) -> nd:
    """
    Generate RGB color values to represent values of a complex array.

    The complex values are mapped to HSV colorspace and then converted to RGB. Hue represents phase and Value represents
    amplitude. Saturation is set to 1.

    Args:
        array: Array to create RGB values for.
        scale: Scaling factor for the array values. When None, scale = 1/max(abs(array)) is used.
        axis: Array axis to use for the RGB dimension.

    Returns:
        An RGB array representing the complex input array.
    """
    if scale is None:
        scale = 1 / np.max(abs(array))
    h = np.expand_dims(np.angle(array) / (2 * np.pi) + 0.5, axis=axis)
    s = np.ones_like(h)
    v = np.expand_dims(np.abs(array) * scale, axis=axis).clip(min=0, max=1)
    hsv = np.concatenate((h, s, v), axis=axis)
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_field(array, scale: Optional[Union[float, nd]] = None, imshow_kwargs: Optional[Dict] = None):
    """
    Plot a complex array as an RGB image.

    The phase is represented by the hue, and the magnitude by the value, i.e. black = zero, brightness shows amplitude,
    and the colors represent the phase.

    Args:
        array(ndarray): complex array to be plotted.
        scale(float): scaling factor for the magnitude. The final value is clipped to the range [0, 1].
        imshow_kwargs: Keyword arguments for matplotlib's imshow.
    """
    if imshow_kwargs is None:
        imshow_kwargs = {}
    rgb = complex_to_rgb(array, scale)
    plt.imshow(rgb, **imshow_kwargs)


def plot_scatter_field(x, y, array, scale, scatter_kwargs=None):
    """
    Plot complex scattered data as RGB values.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {"s": 80}
    rgb = complex_to_rgb(array, scale, axis=1)
    plt.scatter(x, y, c=rgb, **scatter_kwargs)


def complex_colorbar(scale, width_inverse: int = 15):
    """
    Create an rgb colorbar for complex numbers and return its Axes handle.
    """
    amp = np.linspace(0, 1.01, 10).reshape((1, -1))
    phase = np.linspace(0, 249 / 250 * 2 * np.pi, 250).reshape(-1, 1) - np.pi
    z = amp * np.exp(1j * phase)
    rgb = complex_to_rgb(z, 1)
    ax = plt.subplot(1, width_inverse, width_inverse)
    plt.imshow(rgb, aspect="auto", extent=(0, scale, -np.pi, np.pi))

    # Ticks and labels
    ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi), ("$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"))
    ax.set_xlabel("amp.")
    ax.set_ylabel("phase (rad)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return ax


def complex_colorwheel(
    ax: Axes = None,
    shape: Tuple[int, int] = (100, 100),
    imshow_kwargs: dict = {},
    arrow_props: dict = {},
    text_kwargs: dict = {},
    amplitude_str: str = "A",
    phase_str: str = "$\\phi$",
):
    """
    Create an rgb image for a colorwheel representing the complex unit circle.
    TODO: needs review

    Args:
        ax: Matplotlib Axes.
        shape: Number of pixels in each dimension.
        imshow_kwargs: Keyword arguments for matplotlib's imshow.
        arrow_props: Keyword arguments for the arrows.
        text_kwargs: Keyword arguments for the text labels.
        amplitude_str: Text label for the amplitude arrow.
        phase_str: Text label for the phase arrow.

    Returns:
        rgb_wheel: rgb image of the colorwheel.
    """
    if ax is None:
        ax = plt.gca()

    x = np.linspace(-1, 1, shape[1]).reshape(1, -1)
    y = np.linspace(-1, 1, shape[0]).reshape(-1, 1)
    z = x + 1j * y
    rgb = complex_to_rgb(z, scale=1)
    step_width = 1.5 / shape[1]
    blend = np.expand_dims(slope_step(1 - np.abs(z) - step_width, width=step_width), axis=2)
    rgba_wheel = np.concatenate((rgb, blend), axis=2)
    ax.imshow(rgba_wheel, extent=(-1, 1, -1, 1), **imshow_kwargs)

    # Add arrows with annotations
    ax.annotate(
        "",
        xy=(-0.98 / np.sqrt(2),) * 2,
        xytext=(0, 0),
        arrowprops={"color": "white", "width": 1.8, "headwidth": 5.0, "headlength": 6.0, **arrow_props},
    )
    ax.text(**{"x": -0.4, "y": -0.8, "s": amplitude_str, "color": "white", "fontsize": 15, **text_kwargs})
    ax.annotate(
        "",
        xy=(0, 0.9),
        xytext=(0.9, 0),
        arrowprops={
            "connectionstyle": "arc3,rad=0.4",
            "color": "white",
            "width": 1.8,
            "headwidth": 5.0,
            "headlength": 6.0,
            **arrow_props,
        },
    )
    ax.text(**{"x": 0.1, "y": 0.5, "s": phase_str, "color": "white", "fontsize": 15, **text_kwargs})

    # Hide axes spines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
