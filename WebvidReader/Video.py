import itertools
import torch
import numpy
import time as t
import decord
from decord import VideoReader
from decord import cpu, gpu

decord.bridge.set_bridge('torch')

def read_video_file(path, start=0, end=None, channels_first=False, target_resolution=(426, 240)):
    if target_resolution is not None:
        video = VideoReader(path, ctx=cpu(1), width=target_resolution[0], height=target_resolution[1])
    else:
        video = VideoReader(path, ctx=cpu(1))
    return read_video_object(video, start, end, channels_first)


def read_video_object(video, start=0, end=None, channels_first=False):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = []
    frame = 0
    
    key_indices = video.get_key_indices()
    key_frames = video.get_batch(key_indices)
    
    if channels_first and len(key_frames.shape) == 4:
        key_frames = key_frames.permute(0, 3, 2, 1)
    
    return key_frames

