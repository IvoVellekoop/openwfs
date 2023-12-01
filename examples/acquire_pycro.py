from time import time, sleep
from pycromanager import Acquisition, multi_d_acquisition_events, Core
import matplotlib.pyplot as plt

core = Core()

core.set_property('Camera:scanner', 'Invert', '1')
core.set_property('Camera:scanner', 'Delay', 105)

save_dir = r'C:\LocalData'
save_name = r'Acquisition_test'

with Acquisition(directory=save_dir, name=save_name) as acq:
    events = multi_d_acquisition_events(z_start=75, z_end=85, z_step=1.0)
    # events = multi_d_acquisition_events(num_time_points=10)
    acq.acquire(events)

    dataset = acq.get_dataset()

    starttime = time()
    total_num_of_acqs = 11

    while len(dataset.get_index_keys()) < total_num_of_acqs:
        runtime_s = time() - starttime
        num_of_acqs = len(dataset.get_index_keys())
        if runtime_s > 0.5:
            acqs_per_sec = num_of_acqs / runtime_s
        else:
            acqs_per_sec = 0

        if acqs_per_sec > 0:
            eta = (total_num_of_acqs - num_of_acqs) / acqs_per_sec
        else:
            eta = 1e6

        print(f'Running for {runtime_s:.1f}s, {num_of_acqs}/{total_num_of_acqs} acquisitions, ~{acqs_per_sec:.2f} acqs/s, ETA: {eta:.0f}s')
        sleep(1)

    plt.figure()
    for n in range(total_num_of_acqs+1):
        img_raw = dataset.read_image(z=n)
        plt.imshow(img_raw.astype('float32') - 2**15, vmin=-400, vmax=400)
        plt.title(str(n))
        plt.pause(0.01)
        sleep(0.25)

    pass

