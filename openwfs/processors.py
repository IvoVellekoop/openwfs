import numpy as np
import cv2
from typing import Union, Sequence
from .core import Processor, Detector
from .slm.patterns import disk, gaussian
from .utilities import project, Transform


class SingleRoi(Processor):
    """
    Processor that averages a signal over a region of interest.
    """

    def __init__(self, source, x, y, radius=0.1, mask_type='disk', waist=0.5):
        super().__init__(source, data_shape=(1,))
        self._source = source
        self._x = x
        self._y = y
        self._radius = radius
        self._mask_type = mask_type
        self.mask_type = mask_type  # checks if value is correct
        self._mask = None
        self._waist = waist

    def trigger(self, *args, **kwargs):
        # pass x, y and mask to the fetch function, because they may change after
        # triggering and before fetching
        if self._mask is None:
            d = np.floor(self.radius) * 2.0 + 1.0  # make sure the number of pixels is odd, so (0,0) is the center pixel
            r = np.maximum(2.0 * self.radius / d, 0.1)  # always include at least one pixel
            if self._mask_type == 'disk':
                self._mask = disk(d, r)
            elif self._mask_type == 'gaussian':
                self._mask = gaussian(d, r * self._waist)
            else:  # square
                r = np.rint(self.radius).astype('int32')
                self._mask = np.ones((r, r))

        # compute top-left coordinates of the roi
        offset = (self._mask.shape[0] - 1) / 2
        x = self.x - offset
        y = self.y - offset
        return super().trigger(*args, pos=np.rint((y, x)).astype('int32'), mask=self._mask, **kwargs)

    def _fetch(self, out: Union[np.ndarray, None], image: np.ndarray, pos: np.ndarray,
               mask: np.ndarray) -> np.ndarray:  # noqa
        # Implement the logic to fetch the data for this processor
        # crop top/left
        mask_start = np.maximum(0, -pos)
        image_start = np.maximum(0, pos)
        image_end = np.minimum(image.shape, pos + np.array(mask.shape, dtype='int32'))
        mask_end = image_end - image_start
        mask_cropped = mask[mask_start[0]:mask_end[0], mask_start[1]:mask_end[1]]
        image_cropped = image[image_start[0]:image_end[0], image_start[1]:image_end[1]]
        value = np.sum(image_cropped * mask_cropped) / np.sum(mask_cropped)

        if out is None:
            out = np.empty((1,))
        out[:] = value
        return out  # return an array so that we can store metadata on it

    @property
    def x(self) -> int:
        """x-coordinate of the center of the ROI"""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._mask = None

    @property
    def y(self) -> int:
        """y-coordinate of the center of the ROI"""
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self._mask = None

    @property
    def radius(self) -> float:
        """radius of the ROI in pixels"""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._mask = None

    @property
    def mask_type(self):
        """Type of weighting function to use: 'disk', 'square', 'gaussian'"""
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value):
        if value not in ['disk', 'square', 'gaussian']:
            raise ValueError(f"Unknown mask type {value}")
        self._mask_type = value
        self._mask = None

    @property
    def waist(self):
        return self._waist

    @waist.setter
    def waist(self, value):
        self._waist = value
        self._mask = None


class CropProcessor(Processor):
    """Processor to crop data from the source to some region of interest.

    Works on any number of dimensions.
    If the cropped area extends beyond the size of the source data,
    the data is padded with 'padding_value'

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

    def __init__(self, source: Detector, shape: Union[Sequence[int], None] = None,
                 pos: Union[Sequence[int], None] = None, padding_value=0.0):
        super().__init__(source)
        if shape is None:
            shape = source.data_shape

        self._data_shape = shape or source.data_shape
        self._pos = pos if pos is not None else np.zeros((len(source.data_shape),))
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

    def _fetch(self, out: Union[np.ndarray, None], image: np.ndarray) -> np.ndarray:  # noqa
        src_start = np.maximum(self._pos, 0).astype('int32')
        src_end = np.minimum(self._pos + self._data_shape, image.shape).astype('int32')
        dst_start = np.maximum(-self._pos, 0).astype('int32')
        dst_end = dst_start + src_end - src_start
        src_select = tuple(
            slice(start, end) for (start, end) in zip(src_start, src_end))
        src = image.__getitem__(src_select)
        if any(dst_start != 0) or any(dst_end != self._data_shape):
            dst = np.zeros(self._data_shape) + self._padding_value
            dst_select = tuple(slice(start, end) for (start, end) in zip(dst_start, dst_end))
            dst.__setitem__(dst_select, src)
        else:
            dst = src

        if out is None:
            out = dst
        else:
            out[...] = dst
        return out


class SelectRoi(SingleRoi):
    """
    A detector that allows the user to draw a square using the mouse. Inherits from SingleRoiSquare implementation.

    Args:
        source (object): The data source to process.

    Methods:
        draw_square(): Select a rectangular region of interest (ROI) using the mouse.

    """

    def __init__(self, source):

        super().__init__(source, x=0, y=0, mask_type='square')
        source.trigger()
        tl, br = self.draw_square()

    def draw_square(self):
        """
        Select a rectangular region of interest (ROI) using the mouse.
        Returns the ROI coordinates (top-left and bottom-right corners).
        """
        image = self._source.read()

        roi_pts = []
        win_name = "Select ROI and press c, to redraw press r"
        cv2.namedWindow(win_name)

        # Autoscale the image
        image_norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_pts

            if event == cv2.EVENT_LBUTTONDOWN:
                roi_pts = [(x, y)]

            elif event == cv2.EVENT_LBUTTONUP:
                roi_pts.append((x, y))
                cv2.rectangle(image_norm, roi_pts[0], roi_pts[1], (0, 0, 255), 2)
                cv2.imshow(win_name, image_norm)

        cv2.imshow(win_name, image_norm)
        cv2.setMouseCallback(win_name, mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                roi_pts = []
                image_norm = cv2.normalize(image.copy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_8U)
                cv2.imshow(win_name, image_norm)

            elif key == ord("c"):
                if len(roi_pts) == 2:
                    cv2.destroyWindow(win_name)
                    break

            elif key == 27:
                roi_pts = []
                cv2.destroyWindow(win_name)
                break

        if len(roi_pts) == 2:
            x1, y1 = roi_pts[0]
            x2, y2 = roi_pts[1]
            tl = (min(x1, x2), min(y1, y2))
            br = (max(x1, x2), max(y1, y2))

            self.width = br[0] - tl[0]
            self.height = br[1] - tl[1]
            self.top = tl[1]
            self.left = tl[0]

            return tl, br

        return None, None


class SelectRoiCircle(SingleRoi):
    """
    A detector that allows the user to draw a circle using the mouse. Inherits from SingleRoi implementation.

    Args:
        source (object): The data source to process.

    Methods:
        draw_circle(): Select a circular region of interest (ROI) using the mouse.

    """

    def __init__(self, source):
        super().__init__(source, x=0, y=0)
        source.trigger()
        self.draw_circle()

    def draw_circle(self):
        """
        Select a circular region of interest (ROI) using the mouse.
        Returns the ROI coordinates (center and radius).
        """
        image = self.source.read()

        circle_params = []
        win_name = "Select Circle and press c, to redraw press r"
        cv2.namedWindow(win_name)

        # Autoscale the image
        image_norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        def mouse_callback(event, x, y, flags, param):
            nonlocal circle_params

            if event == cv2.EVENT_LBUTTONDOWN:
                circle_params = [(x, y)]

            elif event == cv2.EVENT_LBUTTONUP:
                radius = int(np.sqrt((x - circle_params[0][0]) ** 2 + (y - circle_params[0][1]) ** 2))
                circle_params.append(radius)
                cv2.circle(image_norm, circle_params[0], radius, (0, 0, 255), 2)
                cv2.imshow(win_name, image_norm)

        cv2.imshow(win_name, image_norm)
        cv2.setMouseCallback(win_name, mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                circle_params = []
                image_norm = cv2.normalize(image.copy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_8U)
                cv2.imshow(win_name, image_norm)

            if key == ord("c"):
                if len(circle_params) == 2:
                    cv2.destroyWindow(win_name)
                    break

            elif key == 27:
                circle_params = []
                cv2.destroyWindow(win_name)
                break

        if len(circle_params) == 2:
            self._x, self._y = circle_params[0]
            self._radius = circle_params[1]
            return circle_params[0], circle_params[1]

        return None, None


class TransformProcessor(Processor):
    def __init__(self, source, transform, **kwargs):
        """
        Performs a 2-D transform of the input data (including shifting, padding, cropping, resampling).

        Args:
            transform: Transform object that describes the transformation from the source
            data_shape (in **kwargs): defaults to source.data_shape
            pixel_size (in **kwargs): defaults to source.pixel_size
        """
        if len(source.data_shape) != 2:
            raise ValueError("Source should produce 2-D data")

        super().__init__(source, **kwargs)
        self.transform = transform

    def _fetch(self, out: Union[np.ndarray, None], source) -> np.ndarray:
        return project(self.data_shape, self.pixel_size, source, self.transform, out)
