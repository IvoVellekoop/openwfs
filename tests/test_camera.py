from ..openwfs.devices import Camera
import pytest

cti_path = R"C:\Program Files\Basler\pylon 7\Runtime\x64\ProducerU3V.cti"


def test_grab():
    cam = Camera(cti_path)
    frame = cam.read()
    assert frame.shape == cam.data_shape
    assert (cam.height, cam.width) == cam.data_shape
    assert (cam.nodes.Height.value, cam.nodes.Width.value) == cam.data_shape


@pytest.mark.parametrize("binning", [1, 2, 4])
@pytest.mark.parametrize("top, left", [(0, 0), (21, 17)])
def test_roi(binning, top, left):
    cam = Camera(cti_path)
    original_shape = cam.data_shape

    # check if binning returns the correct shape
    # take care that the size will be a multiple of the increment,
    # and that setting the binning will round this number down
    cam.binning = binning
    expected_width = (original_shape[1] // binning) // cam.nodes.Width.inc * cam.nodes.Width.inc
    expected_height = (original_shape[0] // binning) // cam.nodes.Height.inc * cam.nodes.Height.inc
    assert cam.data_shape == (expected_height, expected_width)

    # check if setting the ROI works
    new_height = cam.height // 2
    new_width = cam.width // 2
    cam.height = new_height
    cam.width = new_width
    cam.top = top
    cam.left = left

    frame = cam.read()
    assert frame.shape == cam.data_shape
    assert (cam.height, cam.width) == cam.data_shape
    assert (cam.nodes.Height.value, cam.nodes.Width.value) == cam.data_shape

    # the camera is allowed to round up to multiples of 16
    # check if cam.width - new_width is in the range 0 to 15
    assert 0 <= cam.width - new_width < cam.nodes.Width.inc
    assert 0 <= cam.height - new_height < cam.nodes.Height.inc
