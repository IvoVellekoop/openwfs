def calibrate_slm_alignment():
    """This function is used for calibrating the alignment of a pupil-conjugate SLM.
    It performs a calibration procedure to find the mapping between SLM pixels, and points on the SLM.

### step 1: construct SLM object with a patch covering the full screen

### step 2: put different gradients on the SLM and see how much the image shifts in micrometers

### step 3: to find the center