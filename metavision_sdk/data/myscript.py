import os
import cv2
import numpy as np

import torch

# Import of Metavision Machine Learning binding
import metavision_sdk_ml
import metavision_sdk_cv
from metavision_sdk_core import EventBbox
from metavision_core.utils import get_sample
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_core.event_io import EventsIterator
from TemporalBinaryRepresentation import TemporalBinaryRepresentation as TBR
from time import time
import matplotlib.pyplot as plt


def init_event_producer(DELTA_T):
    return EventsIterator(SEQUENCE_FILENAME_RAW, start_ts=0, delta_t=DELTA_T, relative_timestamps=False)

DATA_PATH = '/data/metavision_detection_sample/'

SEQUENCE_FILENAME_RAW = DATA_PATH + '/driving_sample.raw'
# SEQUENCE_FILENAME_RAW = DATA_PATH + '/user01_2022-06-08_12-20-03.raw'

DELTA_T = 10000  # 10 ms
#initialize an iterator to get the sensor size
mv_it = init_event_producer(DELTA_T)
ev_height, ev_width = mv_it.get_size()
print("Dimensions:", ev_width, ev_height)

END_TS = None #2 * 1e6 # process sequence until this timestamp (None to disable)

frame = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

tbr_encoder = TBR(N=8, width=ev_width, height=ev_height, incremental=True, cuda=True)

counter = 0
event_rates = []
all_fps = []
for ev in mv_it:
    print(f'----- Frame {counter} -----')
    ts = mv_it.get_current_time()
    if END_TS and ts > END_TS:
        break
    event_rates.append(ev.size)
    print("Rate : {:.2f}Mev/s".format(ev.size * 1e-6))

    tt = time()
    #BaseFrameGenerationAlgorithm.generate_frame(ev, frame) # colors: No event (B:52, G:37, R:30); (200, 126, 64); (255, 255, 255)

    new_frame = np.zeros((ev_height, ev_width), dtype=bool)
    new_frame[ev['y'], ev['x']] = 1

    tbr_frame = tbr_encoder.incremental_update(new_frame)

    tbr_frame = tbr_frame.cpu().numpy()
    cur_fps = 1/(time() - tt)
    all_fps.append(cur_fps)
    print(f'FPS: {cur_fps}')
    # cv2.imwrite(f'frame_{counter}__.png', tbr_frame*255)
    cv2.imshow('Frame', tbr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter += 1

plt.plot(event_rates, all_fps ,'.')
plt.xlabel('Num. events')
plt.ylabel('FPS')
plt.title(f'Delta T: {DELTA_T/1000}ms')
plt.savefig('evt_fps.png')
plt.show()
