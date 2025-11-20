import aedat
from dv import LegacyAedatFile as AedatFile
import os
import numpy as np

def extract_from_file_ae3(directory, filename):
    file = AedatFile(os.path.join(directory, filename))
    data = [[],[],[],[]]
    step = 0
    for event in file:
        if event.timestamp > step:
            step = event.timestamp
        data[0].append(event.timestamp)
        data[1].append(event.x)
        data[2].append(event.y)
        data[3].append(event.polarity)
    return data

def event_puller_ae3(data):
    time_slice = [[], [],[]]
    timing = data[0][0]
    c = 0
    data_iterator = iter(data[0])
    for i in data_iterator:
        timing = i
        time_slice = [[],[],[]]
        while timing == i:
            # print(time_slice)
            time_slice[0].append(data[1][c])
            time_slice[1].append(data[2][c])
            time_slice[2].append(data[3][c])
            c += 1
            try:
                i = next(data_iterator)
                if timing != i:
                    yield time_slice
            except StopIteration:
                break