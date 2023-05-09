import itertools
import torch
import numpy
import time as t
import decord
import cv2
from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as transforms

decord.bridge.set_bridge('torch')

def read_video_file(path, start=0, end=None, channels_first=False, target_resolution=(426, 240), crop_frames=None, broken_frame_warning_only=True):
    if target_resolution is not None:
        video = VideoReader(path, ctx=cpu(1), width=target_resolution[0], height=target_resolution[1])
    else:
        video = VideoReader(path, ctx=cpu(1))
    return read_video_object(video, start, end, channels_first, crop_frames, broken_frame_warning_only)


def read_video_object(video, start=0, end=None, channels_first=False, crop_frames=None, broken_frame_warning_only=True):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    try:
        video_frames = video.get_batch(range(min(len(video), end)))
    except Exception as e:
        print(f"Warning: Failed read video as batch, going to frame by frame reading. The cause was: {str(e)}")
        video_frames = []
        video.seek(0)
        for i in range(min(len(video), end)):
            try:
                video_frames.append(video.next())   
            except Exception as e:
                if(broken_frame_warning_only):
                    print(f"Warning: Failed to load frame '{i}' of video, frame will be skipped. The cause was: {str(e)}")
                else:
                    raise RuntimeError(f"Error: Failed to load frame '{i}' of video. The cause was: {str(e)}")
        
        video_frames = torch.stack(video_frames)
        
    if crop_frames is not None:
        video_frames = video_frames.permute(0, 3, 2, 1)
        transform = transforms.CenterCrop(crop_frames)
        video_frames = transform(video_frames)
        video_frames = video_frames.permute(0, 3, 2, 1)
    
    if channels_first and len(video_frames.shape) == 4:
        video_frames = video_frames.permute(0, 3, 2, 1)
    elif len(video_frames.shape) == 4:
        video_frames = video_frames.permute(0, 2, 1, 3)
    
    return video_frames

def write_video_object(path, torch_data, channels_first=False, target_resolution=None, fps=25, writing_codec='mp4v'):
    if channels_first:
        torch_data = torch_data.permute(0, 3, 2, 1)
    else:
        torch_data = torch_data.permute(0, 2, 1, 3)
    
    if target_resolution is None:
        target_resolution = (torch_data.shape[2], torch_data.shape[1])
    
    fourcc = cv2.VideoWriter_fourcc(*writing_codec)
    out = cv2.VideoWriter(path, fourcc, fps, target_resolution)
    for i in range(torch_data.shape[0]):
        out.write(numpy.uint8(torch_data[i].numpy()))
        
    out.release()

