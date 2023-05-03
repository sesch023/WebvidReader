import itertools
import cv2
import torch
import numpy


def read_video_file(path, start=0, end=None, channels_first=False, target_resolution=(426, 240)):
    video = cv2.VideoCapture(path)
    return read_video_object(video, start, end, channels_first)


def read_video_object(video, start=0, end=None, channels_first=False, target_resolution=(426, 240)):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = []
    frame = 0

    while video.isOpened() and frame <= end:
        ret, frame_data = video.read()
        if not ret:
            break
        
        if target_resolution is not None and frame_data.shape[0] != target_resolution[1] and frame_data.shape[1] != target_resolution[0]:
            frame_data = cv2.resize(frame_data, target_resolution)
        
        if frame >= start:
            video_frames.append(frame_data)

        frame += 1

    video.release()
    
    video = numpy.array(video_frames)
    
    if channels_first and len(video.shape) == 4:
        video = video.transpose(0, 3, 2, 1)
    
    return video

