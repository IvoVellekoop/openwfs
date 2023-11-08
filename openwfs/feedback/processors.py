import numpy as np
import cv2
from typing import Union
from ..core import Processor, DataSource
from ..slm.patterns import disk


class SingleRoi(Processor):
    """
    Processor that returns the average of a circle specified by a certain x, y location & radius. Will return
    the value of the x, y location if radius is 0.

    Args:
        source (object): The data source to process.

    Attributes:
        x (int): x-coordinate of the center of the ROI.
        y (int): y-coordinate of the center of the ROI.
        radius (float): Radius of the ROI in pixels.

    Returns:
        float: The average value of the pixels within the specified ROI.

    Methods:
        read_circle(): Read and return the image data within the specified circular ROI.

    """

    def __init__(self, source, x, y, radius=0.1):
        super().__init__(source, data_shape=(1,))
        self._x = x
        self._y = y
        self._radius = radius
        self._mask = None

    def trigger(self, *args, **kwargs):
        # pass x, y and mask to the fetch function, because they may change after
        # triggering and before _fetch-ing
        if self._mask is None:
            d = np.floor(self.radius) * 2.0 + 1.0  # make sure number of pixels is odd, so (0,0) is the center pixel
            r = np.maximum(2.0 * self.radius / d, 0.1)  # always include at least one pixel
            self._mask = disk(d, r)

        # compute top-left coordinates of the roi
        offset = (self._mask.shape[0] - 1) / 2
        x = self.x - offset
        y = self.y - offset
        return super().trigger(*args, pos=np.rint((y, x)).astype('int32'), mask=self._mask, **kwargs)

    def _fetch(self, out: Union[np.ndarray, None], image: np.ndarray, pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
        """x-coordinate of center of the ROI"""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._mask = None

    @property
    def y(self) -> int:
        """y-coordinate of center of the ROI"""
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


class CropProcessor(Processor):
    """Processor to crop 2D data from the source to some region of interest.

    Args:
        source (object): The data source to process.
        width (int): Width of the cropped region (default is None: use the full width of the source).
        height (int): Height of the cropped region (default is None: use the full width of the source).
        left (int): Leftmost column of the cropped region (default is 0).
        top (int): Topmost row of the cropped region (default is 0).
    """

    def __init__(self, source: DataSource, width=None, height=None, left=0, top=0):
        if len(source.data_shape) != 2:
            raise ValueError("`source` must produce 2-D data.")
        super().__init__(source)
        self._top = top
        self._left = left
        self._data_shape = (height or source.data_shape[0], width or source.data_shape[1])

    @property
    def left(self) -> int:
        return self._left

    @left.setter
    def left(self, value: int):
        self._left = value

    @property
    def right(self) -> int:
        return self.left + self.width

    @right.setter
    def right(self, value: int):
        self.width = value - self.left

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @bottom.setter
    def bottom(self, value: int):
        self.height = value - self.top

    @property
    def width(self) -> int:
        return self._data_shape[1]

    @width.setter
    def width(self, value: int):
        self._data_shape = (self._data_shape[0], value)

    @property
    def height(self) -> int:
        return self._data_shape[0]

    @height.setter
    def height(self, value: int):
        self._data_shape = (value, self._data_shape[1])

    def _fetch(self, out: Union[np.ndarray, None], image: np.ndarray) -> np.ndarray:  # noqa
        # top left corner after padding (becomes 0,0 if it was negative)
        left = np.maximum(0, self._left)
        top = np.maximum(0, self._top)

        # bottom right corner after padding
        bottom = self.bottom
        right = left + self.width

        # compute the amount of padding on all sides bottom-right side
        tpad = top - self._top
        lpad = left - self._left
        bpad = -np.minimum(0, bottom - (image.shape[0] + tpad))
        rpad = -np.minimum(0, right - (image.shape[1] + lpad))
        if tpad != 0 or lpad != 0 or bpad != 0 or rpad != 0:
            image = np.pad(image, pad_width=((tpad, bpad), (lpad, rpad)))

        if out is None:
            out = image[top:bottom, left:right]
        else:
            out[...] = image[top:bottom, left:right]
        return out


class SingleRoiSquare(Processor):
    """
    Square Roi image processor that returns the average of a square specified by a certain width & height.

    Args:
        source (object): The data source to process.
        width (int): Width of the square ROI.
        height (int): Height of the square ROI.
        left (int): Leftmost column of the square ROI (default is 0).
        top (int): Topmost row of the square ROI (default is 0).

    Attributes:
        width (int): Width of the square ROI.
        height (int): Height of the square ROI.
        top (int): Topmost row of the square ROI.
        left (int): Leftmost column of the square ROI.

    Returns:
        float: The average value of the pixels within the specified square ROI.

    Methods:
        read_square(): Read and return the image data within the specified square ROI.

    """

    def __init__(self, source, width=None, height=None, left=0, top=0):
        super().__init__(CropProcessor(source, width=width, height=height, left=left, top=top))

    @property
    def data_shape(self):
        return (1,)

    def read_square(self):
        image = super().read()
        return image[self.top:self.top + self.height, self.left:self.left + self.width]

    def read(self):
        return np.mean(super().read())


class SelectRoiSquare(SingleRoiSquare):
    """
    A detector that allows the user to draw a square using the mouse. Inherits from SingleRoiSquare implementation.

    Args:
        source (object): The data source to process.

    Methods:
        draw_square(): Select a rectangular region of interest (ROI) using the mouse.

    """

    def __init__(self, source):

        super().__init__(source)
        source.trigger()
        tl, br = self.draw_square()

    def draw_square(self):
        """
        Select a rectangular region of interest (ROI) using the mouse.
        Returns the ROI coordinates (top-left and bottom-right corners).
        """
        image = self.source.read()

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
