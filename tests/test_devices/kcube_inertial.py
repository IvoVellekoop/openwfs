# Caution: This test file moves KCubeInertial stages. Ensure that the stages are clear of any obstructions before running these tests. 
import pytest
import openwfs.devices as ow_d
import astropy.units as u
import numpy as np


@pytest.fixture
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

    acc = 20000 / u.s**2 * np.ones(4)
    vel = 500 / u.s * np.ones(4)
    stage.acceleration = acc
    stage.velocity = vel

    assert np.allclose(stage.velocity, vel)
    assert np.allclose(stage.acceleration, acc)


@pytest.mark.parametrize("pair_channels", [True, False])
def test_position_movement(stage, pair_channels):
    """Test position setting and movement."""
    stage.pair_channels = pair_channels

    acc = 20000 / u.s**2 * np.ones(4)
    vel = 500 / u.s * np.ones(4)
    stage.acceleration = acc
    stage.velocity = vel

    p_i = np.ones(4) * 10
    stage.position = np.zeros(4) * 10
    stage.stop()
    stage.wait()
    stage.position = np.ones(4) * 10
    stage.wait()

    assert np.allclose(stage.position, p_i)


@pytest.mark.parametrize("pair_channels", [True, False])
def test_position_relative(stage, pair_channels):
    """Test relative position movement."""
    stage.pair_channels = pair_channels

    acc = 20000 / u.s**2 * np.ones(4)
    vel = 500 / u.s * np.ones(4)
    stage.acceleration = acc
    stage.velocity = vel

    p_f = np.array([10, 10, 10, 10]) * 10
    stage.position = p_f
    stage.wait()

    delta = np.array([10, -10, 10, -10])
    stage.move_by(delta)
    stage.wait()

    assert np.allclose(stage.position, p_f + delta)
