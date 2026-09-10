# Caution: This test file moves KCubeInertial stages. Ensure that the stages are clear of any obstructions before running these tests.
import pytest
import openwfs.devices as ow_d
import astropy.units as u
import numpy as np
import os.path

kinesis_folder = r"C:\Program Files\Thorlabs\Kinesis"
if not os.path.isdir(kinesis_folder):
    pytest.skip("Kinesis not found. Skipping tests.", allow_module_level=True)


@pytest.fixture(scope="module")
def stage():
    """Fixture to create KCubeInertial stage instance for testing."""
    stage_instance = ow_d.KCubeInertial()
    yield stage_instance
    del stage_instance


@pytest.mark.parametrize("pair_channels", [True, False])
def test_pair_channels(stage, pair_channels):
    """Test setting pair_channels mode."""
    stage.pair_channels = pair_channels
    assert stage.pair_channels == pair_channels


@pytest.mark.parametrize("pair_channels", [True, False])
def test_velocity_and_acceleration(stage, pair_channels):
    """Test setting velocity and acceleration."""
    stage.pair_channels = pair_channels

    old_acc = stage.acceleration
    old_vel = stage.velocity

    acc = 20000 / u.s**2 * np.ones(4)
    vel = 500 / u.s * np.ones(4)
    stage.acceleration = acc
    stage.velocity = vel

    assert np.allclose(stage.velocity, vel)
    assert np.allclose(stage.acceleration, acc)

    stage.velocity = old_vel
    stage.acceleration = old_acc

    assert np.allclose(stage.velocity, old_vel)
    assert np.allclose(stage.acceleration, old_acc)


@pytest.mark.parametrize("pair_channels", [True, False])
def test_position_movement(stage, pair_channels):
    """Test position setting and movement."""
    stage.pair_channels = pair_channels

    old_pos = stage.position.copy()
    p_i = old_pos + 100
    stage.position = p_i
    stage.wait()
    assert np.allclose(stage.position, p_i)

    stage.position = old_pos
    stage.wait()
    assert np.allclose(stage.position, old_pos)


@pytest.mark.parametrize("pair_channels", [True, False])
def test_position_relative(stage, pair_channels):
    """Test relative position movement."""
    stage.pair_channels = pair_channels

    p_i = stage.position.copy()
    delta = np.array([10, -10, 10, -10])
    stage.move_by(delta)
    stage.wait()

    assert np.allclose(stage.position, p_i + delta)

    stage.position = p_i
    stage.wait()
    assert np.allclose(stage.position, p_i)
