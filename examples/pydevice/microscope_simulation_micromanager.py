import set_path  # noqa - needed for setting the module search path to find openwfs
import numpy as np
from openwfs.algorithms import FourierDualReference
from openwfs.algorithms.utilities import WFSController
from openwfs.processors import SingleRoi
from openwfs.simulation import Microscope, MockSource, MockSLM
import skimage
import astropy.units as u

"""
This script sets up a simulated adaptive optics system using a mock phase aberration and a simple point source image. 
It simulates the correction of aberrations in a microscopic imaging system through wavefront sensing and shaping.

Variables:
- numerical_aperture: Defines the NA of the system
- aberration_phase: Represents the phase aberrations introduced in the system, generated from a mock camera image and 
    scaled to span a 2π phase range.
- aberration: A MockSource object representing the aberration in the fourier plane. N
- img: A mock image with a central bright spot and several other dimmer points, represented as a 512x512 array.
- signal_location: The coordinates of the primary signal in the mock image.
- src: A MockSource object representing the source image.
- slm: A MockSLM object representing the spatial light modulator.
- sim: A Microscope object simulating the imaging system.
- cam: A camera object representing the detector.
- roi_detector: A SingleRoi object defining a region of interest on the detector for targeted measurements.
- controller: Object to control WFS from MicroManager.
- devices: A dict object that PyDevice reads to access the objects from MicroManager.
"""

numerical_aperture = 1.0
aberration_phase = skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)

img = np.zeros((512, 512), dtype=np.int16)
signal_location = (256, 256)
img[signal_location] = 100
img[23, 70] = 50
img[300, 10] = 50
img[50, 300] = 30
img[400, 20] = 20

src = MockSource(img, 400 * u.nm)

slm = MockSLM(shape=(1000, 1000))

sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=numerical_aperture,
                 aberrations=aberration,
                 wavelength=800 * u.nm)

cam = sim.get_camera(analog_max=10)
roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point
alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=(1000, 1000), k_angles_min=-3, k_angles_max=3,
                           phase_steps=3)
controller = WFSController(alg)

devices = {
    'cam': cam,
    'wfs_controller': controller,
    'slm': slm,
    'stage': sim.xy_stage,
    'microscope': sim,
    'wfs': alg}

# === Uncomment for debugging === #
# controller.wavefront = WFSController.State.FLAT_WAVEFRONT
# before = roi_detector.read()
# controller.wavefront = WFSController.State.SHAPED_WAVEFRONT
# after = roi_detector.read()
# imshow(controller._optimized_wavefront)
# print(after / before)
#
# print(f'SNR: {controller.snr:.5f}')
#
# controller.test_wavefront = 1
# print(f'Feedback enhancement: {controller.feedback_enhancement}')

pass
