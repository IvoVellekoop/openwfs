import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "meadowlark_hdmi: Meadowlark blink for HDMI SLM")
    config.addinivalue_line("markers", "kinesis_inertial: Thorlabs Kinesis device control")
    config.addinivalue_line("markers", "camera: Camera device control")
