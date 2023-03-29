import itertools
import cv2
import torch


def read_video_file(path, start=0, end=None):
    video = cv2.VideoCapture(path)
    return read_video_object(video, start, end)


def read_video_object(video, start=0, end=None):
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

    print(video_frames)
    video.release()
    return torch.tensor(video_frames)

