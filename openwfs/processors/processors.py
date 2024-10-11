from typing import Sequence, Optional

import cv2
import numpy as np
from astropy.units import Quantity

from ..core import Processor, Detector
from ..utilities import project, Transform
from ..utilities.patterns import disk, gaussian


class Roi:
    """
    Represents a Region of Interest (ROI) to compute a (weighted) average over.

    This class defines an ROI with specified properties such as coordinates,
    radius, mask type, and parameters specific to the mask type.
    """

    def __init__(self, pos, radius=0.1, mask_type: str = "disk", waist=None, source_shape=None):
        """
        Initialize the Roi object.

        Args:
            pos (int, int): y,x coordinates of the center of the ROI, measured in pixels from the top-left corner.
                when omitted, the default value of source_shape // 2 is used.
                note: non-integer positions for the ROI are currently not supported.
            radius (float): Radius of the ROI. Default is 0.1.
            mask_type: Type of the mask.
                Options are 'disk' (default), 'gaussian', or 'square'.
            waist (float): Defines the width of the Gaussian distribution in pixels.
                Default is 0.5 * radius.
            source_shape (int, int): Shape of the source image.
                Used to compute a default value for `pos`, and to check if the ROI is fully inside the image.
        """
        if pos is None:
            pos = (source_shape[0] // 2, source_shape[1] // 2)
        if (
            round(pos[0] - radius) < 0
            or round(pos[1] - radius) < 0
            or source_shape is not None
            and (round(pos[0] + radius) >= source_shape[0] or round(pos[1] + radius) >= source_shape[1])
        ):
            raise ValueError("ROI does not fit inside source image")

        self._pos = pos
        self._radius = radius
        self._mask_type = mask_type
        self._waist = waist if waist is not None else radius * 0.5
        self._mask = None
        self._mask_sum = 0.0

    @property
    def pos(self) -> tuple[int, int]:
        return self._pos

    @pos.setter
    def pos(self, value: tuple[int, int]):
        self._pos = value
        self._mask = None  # need to re-compute mask

    @property
    def x(self) -> int:
        """x-coordinate of the center of the ROI, relative to the center of the image."""
        return self.pos[1]

    @x.setter
    def x(self, value: int):
        self.pos = (self.pos[0], int(value))

    @property
    def y(self) -> int:
        """y-coordinate of the center of the ROI, relative to the center of the image."""
        return self.pos[0]

    @y.setter
    def y(self, value: int):
        self.pos = (int(value), self.pos[1])

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = float(value)
        self._mask = None  # need to re-compute mask

    @property
    def waist(self) -> float:
        return self._waist

    @waist.setter
    def waist(self, value: float):
        self._waist = float(value)
        self._mask = None  # need to re-compute mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @waist.setter
    def waist(self, value: str):
        if value not in ["disk", "gaussian", "square"]:
            raise ValueError("mask_type must be 'disk', 'gaussian', or 'square'")
        self._mask_type = value
        self._mask = None  # need to re-compute mask

    def apply(self, image: np.ndarray, order: float = 1.0):
        """
        Applies the mask to the frame data by computing the weighted average.
        Optionally, the image data can be raised to a power before the mask is applied.
        This is useful for simulating multi-photon excitation, or for computing
        weighted contrast over the mask.

        Args:
            image (np.ndarray): The source image data.
            order (float): The order of the mask. Default is 1.0.
        """

        # if any of the variables changed, we need to re-compute the mask
        if self._mask is None:
            # clip the radius so that it corresponds to at least 1 pixel
            r = np.maximum(self._radius, 0.1)

            # for circular masks, always use an odd number of pixels so that we have a clearly
            # defined center.
            # for square masks, instead use the actual size
            if self.mask_type == "disk":
                d = round(self._radius) * 2 + 1
                self._mask = disk(d, r)
            elif self.mask_type == "gaussian":
                d = round(self._radius) * 2 + 1
                self._mask = gaussian(d, self._waist)
            else:  # square
                d = round(self._radius * 2.0)
                self._mask = np.ones((d, d))

            self._mask_sum = np.sum(self._mask)

        image_start = np.array(self.pos) - int(0.5 * self._mask.shape[0] - 0.5)
        image_cropped = image[
            image_start[0] : image_start[0] + self._mask.shape[0],
            image_start[1] : image_start[1] + self._mask.shape[1],
        ]

        if image_cropped.shape != self._mask.shape:
            raise ValueError(
                f"ROI is larger than the possible area. ROI shape: {self._mask.shape}, "
                + f"Cropped image shape: {image_cropped.shape}"
            )

        if order != 1.0:
            image_cropped = np.power(image_cropped, order)

        return np.sum(image_cropped * self._mask) / self._mask_sum


class MultipleRoi(Processor):
    """
    Processor that averages signals over multiple regions of interest (ROIs).
    """

    def __init__(self, source, rois: Sequence[Roi], multi_threaded: bool = True):
        """
        Initialize the MultipleRoi processor with a source and multiple ROIs.
        Note: changing parameters of the ROIs between triggering and fetching causes a race condition.
        Args:
            source (Detector): Source detector object to process the data from.
            rois (Sequence[Roi]): Sequence of Roi objects defining the regions of interest.
        """
        self._rois = np.array(rois)
        self._source = source
        super().__init__(source, multi_threaded=multi_threaded)

    def _fetch(self, image: np.ndarray) -> np.ndarray:  # noqa
        """
        Fetches and processes the data for each ROI from the image.

        This method crops the image according to each ROI's position and
        calculates the average value within the ROI. If an ROI is larger than
        the possible area of the image, a ValueError is raised.

        Args:
            image (np.ndarray): The source image data.

        Returns:
            np.ndarray: Array containing the processed data for each ROI.
        """

        def apply_mask(mask: Roi):
            return mask.apply(image)

        return np.vectorize(apply_mask)(self._rois)

    @property
    def data_shape(self):
        return self._rois.shape

    @property
    def pixel_size(self) -> None:
        """Returns None, since the elements in the output of the MultipleRoi processor do not have a physical size."""
        return None


class SingleRoi(MultipleRoi):
    def __init__(
        self,
        source,
        pos=None,
        radius=0.1,
        mask_type: str = "disk",
        waist=0.5,
        multi_threaded: bool = True,
    ):
        """
        Processor that averages a signal over a single region of interest (ROI).

        Args:
            source (Detector): Source detector object to process the data from.
            pos (int, int): y,x coordinates of the center of the ROI, measured in pixels from the top-left corner.
                when omitted, the default value of source.data_shape // 2 is used.
                note: non-integer positions for the ROI are currently not supported.
            radius (float): Radius of the ROI in pixels. Default is 0.1.
            mask_type: Type of the mask. Options are 'disk', 'gaussian', or 'square'. Default is 'disk'.
            waist (float): Defines the width of the Gaussian distribution. Default is 0.5.
        """
        single_roi = Roi(pos, radius, mask_type, waist, source.data_shape)
        rois = np.array([single_roi]).reshape(())
        super().__init__(source, rois=rois, multi_threaded=multi_threaded)
        self.__dict__.update(single_roi.__dict__)


class CropProcessor(Processor):
    """Processor to crop data from the source to some region of interest.

    Works on any number of dimensions.
    If the cropped area extends beyond the size of the source data,
    the data is padded with 'padding_value'
    """

    def __init__(
        self,
        source: Detector,
        shape: Optional[Sequence[int]] = None,
        pos: Optional[Sequence[int]] = None,
        padding_value=0.0,
        multi_threaded: bool = False,
    ):
        """

        Args:
            source (object): The data source to process.
            shape (tuple): Size of the cropped region (this is data_shape property)
                default is None: use the full size of the source.
                may be a tuple holding one or more None values.
                These values are then replaced by the size of the source in that dimension.
            pos (tuple): Coordinates of the start of the cropped region.
                For 2-D data, this is the top-left corner.
            padding_value (float): Value to use if the cropped area extends beyond the original data.
        """
        super().__init__(source, multi_threaded=multi_threaded)
        self._data_shape = tuple(shape) if shape is not None else source.data_shape
        self._pos = np.array(pos) if pos is not None else np.zeros((len(self.data_shape),), dtype=int)
        self._padding_value = padding_value

    @property
    def pos(self) -> tuple:
        """Start ('top-left' corner) of the cropped region."""
        return tuple(self._pos)

    @pos.setter
    def pos(self, value):
        self._pos = np.array(value, ndmin=1)

    @property
    def data_shape(self) -> tuple:
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._data_shape = tuple(np.array(value, ndmin=1))

    def _fetch(self, image: np.ndarray) -> np.ndarray:  # noqa
        """
        Args:
            image (ndarray): source image

        Returns: the out array containing the cropped image.

        """
        src_start = np.maximum(self._pos, 0).astype("int32")
        src_end = np.minimum(self._pos + self._data_shape, image.shape).astype("int32")
        dst_start = np.maximum(-self._pos, 0).astype("int32")
        dst_end = dst_start + src_end - src_start
        src_select = tuple(slice(start, end) for (start, end) in zip(src_start, src_end))
        src = image.__getitem__(src_select)
        if any(dst_start != 0) or any(dst_end != self._data_shape):
            dst = np.zeros(self._data_shape) + self._padding_value
            dst_select = tuple(slice(start, end) for (start, end) in zip(dst_start, dst_end))
            dst.__setitem__(dst_select, src)
        else:
            dst = src

        return dst


def select_roi(source: Detector, mask_type: str):
    """
    Opens a window that allows the user to select a region of interest.
    """
    if mask_type not in ["disk", "gaussian", "square"]:
        raise ValueError("mask_type must be 'disk', 'gaussian', or 'square'")

    image = cv2.normalize(
        source.read(),
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    title = "Select ROI and press c to continue or ESC to cancel"
    cv2.namedWindow(title)
    cv2.imshow(title, image)

    roi_start = np.array((0, 0))
    roi_size = 0.0

    def mouse_callback(event, x, y, flags, _param):
        nonlocal roi_start, roi_size, image
        if event == cv2.EVENT_LBUTTONDOWN:  # mouse down: select start
            roi_start = np.array((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and cv2.EVENT_FLAG_LBUTTON & flags:
            roi_size = np.minimum(x - roi_start[0], y - roi_start[1])
            rect_image = image.copy()
            if mask_type == "square":
                cv2.rectangle(rect_image, roi_start, roi_start + roi_size, (0.0, 0.0, 255.0), 2)
            else:
                cv2.circle(
                    rect_image,
                    roi_start + roi_size // 2,
                    abs(roi_size) // 2,
                    (0.0, 0.0, 255.0),
                    2,
                )
            cv2.imshow(title, rect_image)

    cv2.setMouseCallback(title, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if roi_size is not None:
                cv2.destroyWindow(title)
                if roi_size < 0.0:
                    roi_start = roi_start + roi_size
                    roi_size = -roi_size
                return Roi(pos=(roi_start[1], roi_start[0]), radius=0.5 * roi_size)

        elif key == 27:
            cv2.destroyWindow(title)
            return None


class TransformProcessor(Processor):
    """
    Performs a 2-D transform of the input data (including shifting, padding, cropping, resampling).

    By default, the output shape and pixel_size are the same as the input shape.
    If desired, explicit values can be provided for the output shape and pixel_size. The unit of the pixel_size
    should match the unit of the input data after applying the transform.
    """

    def __init__(
        self,
        source: Detector,
        transform: Transform = None,
        data_shape: Optional[Sequence[int]] = None,
        pixel_size: Optional[Quantity] = None,
        multi_threaded: bool = True,
    ):
        """

        Args:
            transform: Transform object that describes the transformation from the source to the target image
            data_shape: Shape of the output. If omitted, the shape of the input data is used.
            multi_threaded: Whether to perform processing in a worker thread.
        """
        if (data_shape is not None and len(data_shape) != 2) or len(source.data_shape) != 2:
            raise ValueError("TransformProcessor only supports 2-D data")
        if transform is None:
            transform = Transform()

        # check if input and output pixel sizes are compatible
        dst_unit = transform.destination_unit(source.pixel_size.unit)
        if pixel_size is not None and not pixel_size.unit.is_equivalent(dst_unit):
            raise ValueError("Pixel size unit does not match the unit of the transformed data")
        if pixel_size is None and not source.pixel_size.unit.is_equivalent(dst_unit):
            raise ValueError(
                "The transform changes the unit of the coordinates." " An output pixel_size must be provided."
            )

        self.transform = transform
        super().__init__(source, multi_threaded=multi_threaded)
        self._pixel_size = pixel_size
        self._data_shape = data_shape

    @property
    def data_shape(self):
        return self._data_shape if self._data_shape is not None else super().data_shape

    @property
    def pixel_size(self) -> Optional[Quantity]:
        return self._pixel_size if self._pixel_size is not None else super().pixel_size

    def _fetch(self, source: np.ndarray) -> np.ndarray:  # noqa
        """
        Args:
            source (Detector): A Detector object as described in openwfs.core.Detector

        Returns: ndarray that has been transformed
        TODO: Fix and add test, or remove
        """
        return project(
            source,
            transform=self.transform,
            out_shape=self.data_shape,
            out_extent=self.extent,
        )
