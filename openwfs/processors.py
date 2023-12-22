import numpy as np
import cv2
from typing import Union, Sequence
from .core import Processor, Detector
from .slm.patterns import disk, gaussian
from .utilities import project, Transform
from typing import List, Union


class Roi:
    """
    Represents a Region of Interest (ROI) for image processing. It's used by MultipleRoi.

    This class defines an ROI with specified properties such as coordinates,
    radius, mask type, and parameters specific to the mask type.
    It supports different types of masks like 'disk', 'gaussian', or 'square'.
    """

    def __init__(self, x, y, radius=0.1, mask_type='disk', waist=0.5):
        """
        Initialize the Roi object.

        Args:
            x (int): X-coordinate of the center of the ROI, relative to the center of the image.
            y (int): Y-coordinate of the center of the ROI, relative to the center of the image.
            radius (float): Radius of the ROI. Default is 0.1.
            mask_type (str): Type of the mask. Options are 'disk', 'gaussian', or 'square'. Default is 'disk'.
            waist (float): Defines the width of the Gaussian distribution. Default is 0.5.
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.mask_type = mask_type
        self.mask = None
        self.waist = waist

        # Initialize mask based on the mask type
        self.initialize_mask()

    def initialize_mask(self):
        """
        Initializes the mask based on the specified mask type and parameters.
        """
        d = int(np.floor(self.radius) * 2.0 + 1)
        r = (2.0 * self.radius / d) + 0.0001

        if self.mask_type == 'disk':
            self.mask = disk(d, r)
        elif self.mask_type == 'gaussian':
            self.mask = gaussian(d, r * self.waist)
        else:  # square
            r = int(np.rint(self.radius))
            self.mask = np.ones((r, r))


class MultipleRoi(Processor):
    """
    Processor that averages signals over multiple regions of interest (ROIs).
    """

    def __init__(self, source, rois: List[Roi]):
        """
        Initialize the MultipleRoi processor with a source and multiple ROIs.

        Args:
            source (Detector): Source detector object to process the data from.
            rois (List[Roi]): List of Roi objects defining the regions of interest.
        """
        super().__init__(source, data_shape=(len(rois),))
        self._source = source
        self.rois = rois

    def trigger(self, *args, **kwargs):
        """
        Trigger the processing of the source data with the defined ROIs.

        This method computes the positions for each ROI based on their
        coordinates and prepares the processor for fetching the processed data.

        Args:
            *args & **kwargs: optional input parameters for the source detector
        """
        positions = []
        for roi in self.rois:
            roi.initialize_mask()

            # Compute top-left coordinates of each roi
            offset = (roi.mask.shape[0] - 1) / 2
            x = roi.x - offset
            y = roi.y - offset
            positions.append(np.rint((x, y)).astype('int32'))

        return super().trigger(*args, positions=positions, **kwargs)

    def _fetch(self, out: Union[np.ndarray, None], image: np.ndarray, positions: List[np.ndarray]) -> np.ndarray:
        """
        Fetches and processes the data for each ROI from the image.

        This method crops the image according to each ROI's position and
        calculates the average value within the ROI. If an ROI is larger than
        the possible area of the image, a ValueError is raised.

        Args:
            out (Union[np.ndarray, None]): Optional output array to store the processed data.
            image (np.ndarray): The source image data.
            positions (List[np.ndarray]): List of positions for each ROI.

        Returns:
            np.ndarray: Array containing the processed data for each ROI.
        """
        if out is None:
            out = np.empty((len(self.rois),))

        for idx, (roi, pos) in enumerate(zip(self.rois, positions)):
            # Crop image
            image_start = np.array(((image.shape[0] - 1) / 2) + pos, dtype='int32')
            image_end = np.minimum(image.shape, image_start + np.array(roi.mask.shape, dtype='int32'))

            image_cropped = image[image_start[0]:image_end[0], image_start[1]:image_end[1]]

            if roi.mask.shape[0] > image_cropped.shape[0] or roi.mask.shape[1] > image_cropped.shape[1]:
                raise ValueError(
                    f"ROI is larger than the possible area. ROI shape: {roi.mask.shape}, Cropped image shape: {image_cropped.shape}")

            value = np.sum(image_cropped * roi.mask) / np.sum(roi.mask)

            out[idx] = value

        return out


class SingleRoi(MultipleRoi):
    """
    Processor that averages a signal over a single region of interest (ROI).
    """

    def __init__(self, source, x, y, radius=0.1, mask_type='disk', waist=0.5):
        """
        Initialize the SingleRoi processor with a source and a single ROI.

        Args:
            source (Detector): Source detector object to process the data from.
            x (int): X-coordinate of the center of the ROI, relative to the center of the image.
            y (int): Y-coordinate of the center of the ROI, relative to the center of the image.
            radius (float): Radius of the ROI. Default is 0.1.
            mask_type (str): Type of the mask. Options are 'disk', 'gaussian', or 'square'. Default is 'disk'.
            waist (float): Defines the width of the Gaussian distribution. Default is 0.5.
        """
        single_roi = Roi(x, y, radius, mask_type, waist)
        super().__init__(source, rois=[single_roi])
        self.single_roi = single_roi

    @property
    def x(self) -> int:
        """x-coordinate of the center of the ROI, relative to the center of the image."""
        return self.single_roi.x

    @x.setter
    def x(self, value):
        self.single_roi.x = value
        self.single_roi.initialize_mask()

    @property
    def y(self) -> int:
        """y-coordinate of the center of the ROI, relative to the center of the image."""
        return self.single_roi.y

    @y.setter
    def y(self, value):
        self.single_roi.y = value
        self.single_roi.initialize_mask()

    @property
    def radius(self) -> float:
        """radius of the ROI in pixels"""
        return self.single_roi.radius

    @radius.setter
    def radius(self, value):
        self.single_roi.radius = value
        self.single_roi.initialize_mask()

    @property
    def mask_type(self):
        """Type of weighting function to use: 'disk', 'square', 'gaussian'"""
        return self.single_roi.mask_type

    @mask_type.setter
    def mask_type(self, value):
        if value not in ['disk', 'square', 'gaussian']:
            raise ValueError(f"Unknown mask type {value}")
        self.single_roi.mask_type = value
        self.single_roi.initialize_mask()

    @property
    def waist(self):
        """Parameter for the Gaussian mask, defining the width of the Gaussian distribution."""
        return self.single_roi.waist

    @waist.setter
    def waist(self, value):
        self.single_roi.waist = value
        self.single_roi.initialize_mask()


class CropProcessor(Processor):
    """Processor to crop data from the source to some region of interest.

    Works on any number of dimensions.
    If the cropped area extends beyond the size of the source data,
    the data is padded with 'padding_value'
    """

    def __init__(self, source: Detector, shape: Union[Sequence[int], None] = None,
                 pos: Union[Sequence[int], None] = None, padding_value=0.0):
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
        super().__init__(source)
        if shape is None:
            shape = source.data_shape

        self._data_shape = shape or source.data_shape
        self._pos = pos if pos is not None else np.zeros((len(source.data_shape),), dtype=int)
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
        """
        Args:
            out(ndarray) optional numpy array or view of an array that will receive the data
                when present, the data will be stored in `out`, and `out` is returned.
            image (ndarray): source image

        Returns: the out array containing the cropped image.

        """
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
    """

    def __init__(self, source):
        """

        Args:
            source (Detector): Detector object to process the data from.
        """

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

    Methods:
        draw_circle(): Select a circular region of interest (ROI) using the mouse.

    """

    def __init__(self, source):
        """
        Initialising triggers the draw_circle method.

        Args:
            source (Detector): Source detector object to process the data from.
        """
        super().__init__(source, x=0, y=0)
        source.trigger()
        self.draw_circle()

    def draw_circle(self):
        """
        Select a circular region of interest (ROI) using the mouse.
        Returns the ROI coordinates (center and radius).
        """
        image = self._source.read()

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
    """
    Performs a 2-D transform of the input data (including shifting, padding, cropping, resampling).
    """

    def __init__(self, source, transform, **kwargs):
        """

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
        """
        Args:
            out(ndarray) optional numpy array or view of an array that will receive the data
                when present, the data will be stored in `out`, and `out` is returned.
            source (Detector): A Dectector object as described in openwfs.core.Detector

        Returns: ndarray that has been transformed
        """
        return project(self.data_shape, self.pixel_size, source, self.transform, out)
