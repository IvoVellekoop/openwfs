import pytest

from ..openwfs.devices import Camera, safe_import

harvesters = safe_import("harvesters", "harvesters")
if not harvesters:
    pytest.skip(
        "harvesters is required for the Camera module, install with pip install harvesters", allow_module_level=True
    )


cti_path = R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti"


@pytest.fixture
def camera():
    """
    Fixture that returns a Camera object.
    If no camera is found, the test is skipped.
    """
    cameras = Camera.enumerate_cameras(cti_path)
    if len(cameras) == 0:
        pytest.skip("No camera found", allow_module_level=True)
    return Camera(cti_path)


def test_grab(camera):
    frame = camera.read()
    assert frame.shape == camera.data_shape
    assert (camera.height, camera.width) == camera.data_shape
    assert (camera._nodes.Height.value, camera._nodes.Width.value) == camera.data_shape


@pytest.mark.parametrize("binning", [1, 2, 4])
@pytest.mark.parametrize("top, left", [(0, 0), (21, 17)])
def test_roi(camera, binning, top, left):
    original_shape = camera.data_shape

    # check if binning returns the correct shape
    # take care that the size will be a multiple of the increment,
    # and that setting the binning will round this number down
    camera.binning = binning
    expected_width = (original_shape[1] // binning) // camera._nodes.Width.inc * camera._nodes.Width.inc
    expected_height = (original_shape[0] // binning) // camera._nodes.Height.inc * camera._nodes.Height.inc
    assert camera.data_shape == (expected_height, expected_width)

    # check if setting the ROI works
    new_height = camera.height // 2
    new_width = camera.width // 2
    camera.height = new_height
    camera.width = new_width
    camera.top = top
    camera.left = left

    frame = camera.read()
    assert frame.shape == camera.data_shape
    assert (camera.height, camera.width) == camera.data_shape
    assert (camera._nodes.Height.value, camera._nodes.Width.value) == camera.data_shape

    # the camera is allowed to round up to multiples of 16
    # check if camera.width - new_width is in the range 0 to 15
    assert 0 <= camera.width - new_width < camera._nodes.Width.inc
    assert 0 <= camera.height - new_height < camera._nodes.Height.inc
