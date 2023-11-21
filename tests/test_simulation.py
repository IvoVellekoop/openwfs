import matplotlib.pyplot as plt
import pytest
import numpy as np
from ..openwfs.algorithms import StepwiseSequential, BasicFDR, CharacterisingFDR
from ..openwfs.processors import SingleRoi
from ..openwfs.simulation import SimulatedWFS,Microscope,MockCamera,MockSource,MockXYStage,MockSLM
import skimage
from ..openwfs.slm import SLM
from ..openwfs.slm.patterns import tilt,disk
import astropy.units as u

def test_MockCamera_and_SingleRoi():
    """
    The MockCamera is supposed to wrap any Datasource into a base-16-int image source, as MicroManager requires the
    camera object to return the images in that form. It casts the maximum value to the 16-int maximum, signified by the
    hexidecimal 0xFFFF in the default maximum.
    Returns:

    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 39.39 # some random float
    src = MockCamera(MockSource(img, 450 * u.nm))
    roi_detector = SingleRoi(src, x=256, y=256, radius=0) # Only measure that specific point
    assert roi_detector.read() == int(2**16-1) # it should cast the array into some int

def test_Microscope_without_magnification():
    '''
    Attempt to understand how the microscope works. Without any magnification, and the same
    '''
    img = np.zeros((1000, 1000), dtype=np.int16)

    img[256, 256] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    sim = Microscope(source=src, magnification=1, numerical_aperture=1, wavelength=800 * u.nm,
                     camera_pixel_size=400 * u.nm, camera_resolution=(1000,1000))

    assert sim.camera.read()[256,256] == 2**16 -1

def test_Microscope_and_aberration():
    '''
    This test concens the basic effect of casting an aberration or SLM pattern on the backpupil.
    They should aberrate a point source in the image plane.
    '''
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    slm = MockSLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)

    sim = Microscope(source=src, magnification=1,slm=slm.pixels(), numerical_aperture=1, wavelength=800 * u.nm,
                     camera_pixel_size=400 * u.nm, camera_resolution=(1000,1000))

    without_aberration = sim.read()[256,256]
    slm.set_phases(aberrations)
    with_aberration = sim.read()[256,256]
    assert with_aberration < without_aberration


def test_SLM_and_aberration():
    """
    As mentioned in the previous test, casting a pattern on the pupil plane with an SLM and an aberration
    should produce the same effect. We will test that here by projecting two opposing patterns on the pupil plane.
    (Which should do nothing in the image plane)
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    img[256, 256] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    slm = MockSLM(shape=(512, 512))

    aberrations = skimage.data.camera() * ((2 * np.pi) / 255.0)*0
    slm.set_phases(-aberrations)
    aberration = MockSource(aberrations, pixel_size=1.0 / 512 * u.dimensionless_unscaled)

    sim1 = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=1,
                     aberrations=aberration, wavelength=800 * u.nm,camera_pixel_size=400 * u.nm,
                     camera_resolution=(1000,1000))

    sim2 = Microscope(source=src, magnification=1, numerical_aperture=1, wavelength=800 * u.nm,
                      camera_pixel_size=400 * u.nm, camera_resolution=(1000,1000))

    # We correlate the two. Any discrepency between the two matrices should throw an error
    # try putting one of the wavelengths to 800

    a = sim1.read()
    b = sim2.read()
    norm_a = a / np.linalg.norm(a[:])
    norm_b = b / np.linalg.norm(b[:])

    assert abs(np.vdot(norm_a, norm_b)) >= 1



def test_SLM_tilt():
    """
    Display a tilt on the SLM should result in an image plane shift. If the magnification is 1, this should
    correspond to a tilt of 1 pixel for a 2 pi phase shift.
    """
    img = np.zeros((1000, 1000), dtype=np.int16)
    signal_location = (256,256)
    img[signal_location] = 100
    src = MockCamera(MockSource(img, 400 * u.nm))

    slm = MockSLM(shape=(1000,1000))

    sim = Microscope(source=src, slm=slm.pixels(), magnification=1, numerical_aperture=1,
                     wavelength=800 * u.nm,camera_pixel_size=400 * u.nm,
                     camera_resolution=(1000,1000))

    # introduce a tilted pupil plane
    shift = (-24,40)
    slm.set_phases(tilt(1000,shift))

    # Tilt function is in y,x convention, so to get the correct x,y coordinates of the next point:
    new_location = (signal_location[0] + shift[1], signal_location[1] + shift[0])

    assert sim.camera.read()[new_location] == 2**16 - 1

def test_crop():
    """
    Tests how the crop function works in the microscope
    """
    img = np.ones((512,512))
    src = MockSource(img,1/512 * u.dimensionless_unscaled)
    sim = Microscope(source=src, magnification=1, numerical_aperture=1,
                     wavelength=800 * u.nm,camera_pixel_size=400 * u.nm,
                     camera_resolution=(512,512))
    sim._pupil_resolution = 1000

    # So we expect the function to return a full matrix with ones. Unfortunately, the last row in x
    # and y direction it returns are zeros
    assert (np.sum(sim._crop(src.read()))) == 1000*1000
