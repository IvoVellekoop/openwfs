import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "meadowlark_hdmi: Meadowlark blink for HDMI SLM")
    config.addinivalue_line("markers", "kinesis_inertial: Thorlabs Kinesis device control for the inertial stage")
    config.addinivalue_line(
        "markers", "kinesis_motorized_flipper: Thorlabs Kinesis device control for the motorized flipper stage"
    )
    config.addinivalue_line("markers", "camera: Camera device control")
