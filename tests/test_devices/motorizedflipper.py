import openwfs.devices as ow_d

stage = ow_d.MotorizedFilterFlip()

stage.position = True
stage.wait()
stage.position
assert stage.position == 1

stage.position = False
stage.wait()
stage.position
assert stage.position == 0

del stage
