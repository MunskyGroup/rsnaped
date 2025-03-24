# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:06:03 2025

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import linecache
import os
import tracemalloc
import pathlib
import h5py
import time

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def make_random_video(frames):
    return np.random.randint(0,6000,size=(frames,512,512,4),dtype=np.uint16)


def buffer_video_npy(frames, buffersize=5, clear=False):
    frame_count = frames
    i = 0
    
    # delete any previous buffer file
    tmp_path = pathlib.Path('./tmp.bin')
    if tmp_path.exists():
        tmp_path.unlink() 
    
    with open('./tmp.bin', "ab") as f:
        while frame_count > 0:
            if buffersize <= frame_count:
                vid = make_random_video(buffersize)
            else:
                vid = make_random_video(frame_count%buffersize)
            
            np.save(f, vid)
            frame_count -= buffersize
            i += 1
            if i > 1000:
                break
        
    return np.memmap('./tmp.bin', dtype=np.uint16, shape=(frames,512,512,4))

def buffer_video_h5(frames, buffersize=5):
    frame_count = frames
    i = 0
    
    # delete any previous buffer file
    tmp_path = pathlib.Path('./tmp.h5')
    if tmp_path.exists():
        tmp_path.unlink() 
    
    with h5py.File(tmp_path, "w") as f:
        dset = f.create_dataset('current_vid', (frames,512,512,4),
                                    dtype='uint16')
        for i in range(frames):
            dset[i,:,:,:] = make_random_video(1)
    
    return dset

def buffer_video_ndstorage(frames, buffersize=5):
    
    return 


# tracemalloc.start()
# a = []
# snapshot1 = tracemalloc.take_snapshot()
# st = time.time()
# a = make_random_video(1000)
# t = time.time()-st
# snapshot2 = tracemalloc.take_snapshot()
# top_stats = snapshot2.compare_to(snapshot1, 'lineno')
# tracemalloc.stop()
# display_top(snapshot2)
# print('Generated video without buffer in: %s seconds'%str(t))


# print('################ Buffering with NPY ##################')

# tracemalloc.start()
# a = []
# snapshot1 = tracemalloc.take_snapshot()
# st = time.time()
# a = buffer_video_npy(1000)
# t = time.time()-st
# snapshot2 = tracemalloc.take_snapshot()
# top_stats = snapshot2.compare_to(snapshot1, 'lineno')
# tracemalloc.stop()
# display_top(snapshot2)
# print('Generated video with numpy buffer in: %s seconds'%str(t))


print('################ Buffering with H5py ##################')

tracemalloc.start()
a = []
snapshot1 = tracemalloc.take_snapshot()
st = time.time()
a = buffer_video_h5(1000)
t = time.time()-st
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
tracemalloc.stop()
display_top(snapshot2)
print('Generated video with numpy buffer in: %s seconds'%str(t))







