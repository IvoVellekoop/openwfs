import numpy as np
import cv2

from wfs import wavefront_shaping

from fourier import FourierDualRef
import matplotlib.pyplot as plt

from base_device_properties import object_property, bool_property, parse_options

def manual_slm_setup(monitor_id=2, wavelength_nm = 804):
    s = SLM(monitor_id)
    s.set_activepatch(-1)
    s.set_data(0)
    s.update()
    s.set_activepatch(0)
    s.set_data(0)
    s.update()
    s.set_activepatch(1)
    s.set_data(0)
    s.update()
    s.set_activepatch(0)
    s.set_rect([-0.0147, 0.0036, 0.8000, 0.8000])

    # set lut

    return s

def slm_setup(s, wavelength_nm = 804):

    if wavelength_nm <= 400:
        raise ValueError(f'Unexpected wavelength, are you sure you are entering the wavelength in nanometers?')

    s.set_activepatch(0)
    s.set_data(0)
    s.set_activepatch(1)
    s.set_data(0)

    s.set_activepatch(0)
    s.set_rect([-0.0147, 0.0036, 0.8000, 0.8000])

    # # set LUT
    # s.set_activepatch(-2)
    # pats = s.get_activepatch()
    # alpha = 0.2623*wavelength_nm - 23.33
    # LUT = np.arange(0, 255, 1)
    # s.set_data(200)

    s.set_activepatch(0)
    s.update()

# s = SLM(2)
# slm_setup(s)
# s.set_data(np.random.rand(4,4)*256)
# s.update(100)



def select_roi(image):
    """
    Select a rectangular region of interest (ROI) using the mouse.
    Returns the ROI coordinates (top-left and bottom-right corners).
    """
    roi_pts = []
    win_name = "Select ROI"
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


# def take_image():
#     im = single_capture(output_mapping=['Dev2/ao0', 'Dev2/ao1'], input_mapping=['Dev2/ai0'], zoom=[50], delay=[145],
#                         scanpaddingfactor=[1.1],resolution = [200, 200])
#     return im


def single_capt():
    im = take_image()
    return [np.mean(im)]


def point_capt(xrange, yrange,get_feedback):

    im = get_feedback()
    return im[xrange[0]:xrange[1], yrange[0]:yrange[1]]


def make_point_mean(xrange, yrange, get_feedback):
    def point_mean_closure():
        return [np.mean(point_capt(xrange, yrange, get_feedback))]

    return point_mean_closure



def select_point(get_feedback):
    im = get_feedback()
    tl, bl = select_roi(im)
    return [tl[1], bl[1]], [tl[0], bl[0]]

class WfsExperiment:
    def __init__(self,**kwargs):
        self.ranges = False
        self.feedback_set = []
        self.post_process = True
        parse_options(self, kwargs)

    def wfs_procedure(self, algorithm, slm, get_feedback, ranges=False):

        slm_setup(slm, slm._wavelength_nm)

        slm.set_data(0)
        slm.update()

        initial_image = get_feedback()

        if not ranges:
            xrange, yrange = select_point(get_feedback)
        else:
            xrange = ranges[0]
            yrange = ranges[1]

        feedback_func = make_point_mean(xrange, yrange, get_feedback)

        [feedback_set, ideal_wavefronts, t_set] = wavefront_shaping(slm, feedback_func, algorithm,post_process=self.post_process)
        if self.post_process:
            slm.set_data(ideal_wavefronts[:, :, 0])
            slm.update(30)
            im2 = get_feedback
            return ideal_wavefronts[:, :, 0], [xrange, yrange], feedback_set

        else:
            return feedback_set
    def take_image(self):
        self.camera_object.trigger()
        self.camera_object.wait()

        return np.reshape(self.camera_object.image, [self.camera_object._height, self.camera_object._width], order='C')

    def on_execute(self, value = True):
        if value:
            if self.post_process:
                self.optimised_wf, self.range, self.feedback_set = self.wfs_procedure(self.algorithm, self.slm_object, self.take_image, self.ranges)
            else:
                self.feedback_set = self.wfs_procedure(self.algorithm, self.slm_object,self.take_image, self.ranges)
        return value

    def set_optimised_wf(self,value):
        if value:

            self.slm_object.set_data(self.optimised_wf)
            self.slm_object.update(10)

        return value

    def set_flat_wf(self,value):
        if value:
            self.slm_object.set_data(0)
            self.slm_object.update(10)

        return value

    active_slm = False
    slm = None


    algorithm = object_property()
    optimised_wf = None
    slm_object = object_property()
    camera_object = object_property()
    execute = bool_property(default=0, on_update=on_execute)
    show_optimised_wavefront = bool_property(default=0, on_update=set_optimised_wf)
    show_flat_wavefront = bool_property(default=0, on_update=set_flat_wf)


if __name__ == "__main__":
    from openwfs.simulation import SimulatedWFS
    wfs = WfsExperiment()
    sim = SimulatedWFS()
    sim.set_ideal_wf(np.zeros([500,500]))
    sim.E_input_slm = np.ones([500, 500])
    wfs.algorithm = FourierDualRef()
    wfs.ranges = [[250, 251], [250, 251]]
    wfs.algorithm.ky_angles_max = 2
    wfs.algorithm.ky_angles_min = -2
    wfs.algorithm.kx_angles_max = 2
    wfs.algorithm.kx_angles_min = -2
    wfs.algorithm.kx_angles_stepsize=1
    wfs.algorithm.ky_angles_stepsize=1

    wfs.algorithm.build_kspace()
    print(wfs.algorithm.kx_set)
    print(wfs.algorithm.ky_set)

#    wfs.algorithm.set_kspace([-4,2],[7,9])

    wfs.slm_object = sim
    wfs.camera_object = sim

    wfs.execute = 1
    plt.imshow(wfs.optimised_wf)
    plt.show()
    # or you can use wfs.on_execute(), works either way
