import itertools
import cv2
import torch


def read_video_file(path, start=0, end=None, channels_first=False):
    video = cv2.VideoCapture(path)
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

    while video.isOpened() and frame <= end:
        ret, frame_data = video.read()
        if not ret:
            break

        if frame >= start:
            video_frames.append(frame_data)

        frame += 1

    video.release()
    video = torch.tensor(video_frames)
    
    if channels_first:
        video = video.permute(0, 3, 1, 2)
    
    return video

