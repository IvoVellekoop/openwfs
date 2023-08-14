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
    def __init__(self, source, x, y, radius=0.0):
        super().__init__(source)
        self._x = x
        self._y = y
        self._radius = radius

    def read(self):
        image = super().read()
        return image[self.x, self.y]

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

    def read_square(self):
        image = super().read()
        bottom = self.top + self.height
        right = self.left + self.width
        return image[self.top:bottom, self.left:right]

    def read(self):
        
        return int(np.mean(self.read_square()))

class SingleRoiSquare(Processor):
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
    # Inherits from square selection
    def __init__(self, source, width=None, height=None, left=None, top=None):
        if width is height is left is top is None:
            source.trigger()

            tl, br = self.draw_square(source.read())
            width = br[0]-tl[0]
            height = br[1]-tl[1]
            left = tl[1]
            top = tl[0]

        super().__init__(source,width,height,left,top)
        self._data_shape = None
        self._set_shape(width, height)

    def draw_square(self,image):
        """
        Select a rectangular region of interest (ROI) using the mouse.
        Returns the ROI coordinates (top-left and bottom-right corners).
        """
        roi_pts = []
        win_name = "Select ROI and press c"
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
                image_copy = image_norm.copy()
                cv2.imshow(win_name, image_copy)

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
            return tl, br

        return None, None

