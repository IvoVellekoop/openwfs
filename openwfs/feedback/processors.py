import numpy as np
import cv2

class Processor:
    def __init__(self, source):
        self.source = source

    def trigger(self):
        self.source.trigger()

    @property
    def data_shape(self):
        return self.source.data_shape

    @property
    def measurement_time(self):
        return self.source.measurement_time

    @property
    def pixel_size(self):
        return self.source.pixel_size

    def read(self):
        return self.source.read()


class SingleRoi(Processor):
    """
    Processor that returns the average of a circle specified by a certain x,y location & radius. Will return
    the value of the x,y location if radius is 0.
    """
    def __init__(self, source, x, y, radius=0.0):
        super().__init__(source)
        self._x = x
        self._y = y
        self._radius = radius

    def read(self):
        image = super().read()
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (self._x, self._y), int(self._radius), 1, thickness=-1)
        return np.mean(image[mask == 1])

    def read_circle(self):
        image = super().read()
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (self._x, self._y), int(self._radius), 1, thickness=-1)
        return image*mask

    @property
    def data_shape(self):
        return (1,)

    @property
    def x(self) -> int:
        """x-coordinate of center of the ROI"""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> int:
        """y-coordinate of center of the ROI"""
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def radius(self) -> float:
        """radius of the ROI in pixels"""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value


class CropProcessor(Processor):
    def __init__(self, source, width=None, height=None, left=0, top=0):
        super().__init__(source)
        self.left = left
        self.top = top
        if width is None:
            width = source.data_shape[1]
        if height is None:
            height = source.data_shape[0]
        self._data_shape = None
        self._set_shape(width, height)

    def _set_shape(self, width, height):
        ss = self.source.data_shape
        self._data_shape = (np.minimum(ss[0] - self.top, height), np.minimum(ss[1] - self.left, width))

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def width(self) -> int:
        return self.data_shape[1]

    @width.setter
    def width(self, value):
        self._set_shape(value, self.height)

    @property
    def height(self) -> int:
        return self.data_shape[0]

    @height.setter
    def height(self, value):
        self._set_shape(self.width, value)

    def read(self):
        image = super().read()
        bottom = self.top + self.height
        right = self.left + self.width
        return image[self.top:bottom, self.left:right]


class SingleRoiSquare(Processor):
    """
    Square Roi image processor that returns the average of a square specified by a certain width & height.
    """

    def __init__(self, source, width=None, height=None, left=0, top=0):
        super().__init__(source)
        self.left = left
        self.top = top
        if width is None:
            width = source.data_shape[1]
        if height is None:
            height = source.data_shape[0]
        self._data_shape = None
        self._set_shape(width, height)

    def _set_shape(self, width, height):
        ss = self.source.data_shape
        self._shape = (np.minimum(ss[0] - self.top, height), np.minimum(ss[1] - self.left, width))

    @property
    def data_shape(self):
        return (1,)

    @property
    def shape(self):
        return self._shape

    @property
    def width(self) -> int:
        return self._shape[1]

    @width.setter
    def width(self, value):
        self._set_shape(value, self.height)

    @property
    def height(self) -> int:
        return self._shape[0]

    @height.setter
    def height(self, value):
        self._set_shape(self.width, value)

    def read_square(self):
        image = super().read()
        bottom = self.top + self.height
        right = self.left + self.width
        return image[self.top:bottom, self.left:right]

    def read(self):
        
        return np.mean(self.read_square())


class SelectRoiSquare(SingleRoiSquare):
    """
    A detector that allows the user to draw a square using the mouse. Inherits from SingleRoiSquare implementation
    """
    def __init__(self, source):

        super().__init__(source)
        source.trigger()
        self.draw_square()

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

            width = br[0] - tl[0]
            height = br[1] - tl[1]
            self.top = tl[1]
            self.left = tl[0]

            self._set_shape(width, height)
            return tl, br

        return None, None


class SelectRoiCircle(SingleRoi):
    """
    A detector that allows the user to draw a circle using the mouse. Inherits from SingleRoi implementation.
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
