# Caution: This test file moves MotorizedFilterFlip stages. Ensure that the stages are clear of any obstructions before running these tests.
import pytest
import openwfs.devices as ow_d
import os.path

pytestmark = pytest.mark.kinesis_motorized_flipper

kinesis_folder = r"C:\Program Files\Thorlabs\Kinesis"
if not os.path.isdir(kinesis_folder):
    pytest.skip("Kinesis not found. Skipping tests.", allow_module_level=True)


@pytest.fixture(scope="module")
def stage():
    """Fixture to create MotorizedFilterFlip stage instance for testing."""
    stage_instance = ow_d.MotorizedFilterFlip()
    yield stage_instance
    del stage_instance


def test_position_true(stage):
    """Test setting position to True (1)."""
    stage.position = True
    stage.wait()
    assert stage.position == 1


def test_position_false(stage):
    """Test setting position to False (0)."""
    stage.position = False
    stage.wait()
    assert stage.position == 0
