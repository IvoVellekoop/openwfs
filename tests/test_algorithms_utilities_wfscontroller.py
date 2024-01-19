import numpy as np
import skimage
import astropy.units as u

from ..openwfs.algorithms import FourierDualReference
from ..openwfs.algorithms.utilities import WFSController
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import Microscope, MockSource, MockSLM, SimulatedWFS, MockCamera
from ..openwfs.utilities import imshow


def test_wfs_troubleshooter():
    # Define mock hardware
    numerical_aperture = 1.0
    aberration_phase = 0.5 * skimage.data.camera() * ((2 * np.pi) / 255.0) + np.pi
    aberration = MockSource(aberration_phase, extent=2 * numerical_aperture)

    img = np.zeros((256, 256), dtype=np.int16)
    img[128, 128] = 95
    img[70, 70] = 50
    img[40, 40] = 50
    img[70, 40] = 40
    img[40, 70] = 30
    img[128, 70] = 80

    src = MockSource(img, 400 * u.nm)

    slm_shape = (1000, 1000)
    slm = MockSLM(shape=slm_shape)

    sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=numerical_aperture,
                    aberrations=aberration, wavelength=800 * u.nm)

    cam = sim.get_camera(analog_max=100.0, gaussian_noise_std=0.000)
    roi_detector = SingleRoi(cam, radius=0)  # Only measure that specific point
    alg = FourierDualReference(feedback=roi_detector, slm=slm, slm_shape=slm_shape,
                               k_angles_min=-1, k_angles_max=1, phase_steps=3)
    control = WFSController(alg, cam)

    control.troubleshoot()

    # #=== Uncomment for debugging ===#
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.log10(control.read_after_frame_flatwf().clip(1)), vmin=0, vmax=5)
    # plt.title('Flat wavefront')
    # plt.colorbar()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.log10(control.read_after_frame_shapedwf().clip(1)), vmin=0, vmax=5)
    # plt.title(f'Shaped wavefront\nCNR: {control.frame_cnr:.3f}, η_σ: {control.contrast_enhancement:.3f}')
    # plt.colorbar()
    # plt.show()
