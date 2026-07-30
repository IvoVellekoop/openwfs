import openwfs.devices as ow_d
import time

stage = ow_d.MotorizedFilterFlip()

stage.position = 1
stage.wait()
assert stage.position == 1

stage.position = 0
stage.wait()
assert stage.position == 0

del stage
