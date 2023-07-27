import itertools
import torch
import numpy
import time as t
import decord
import cv2
from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as transforms
from einops import rearrange

decord.bridge.set_bridge('torch')

class IllegalStartException(Exception):
    pass

class IllegalFrameException(Exception):
    pass

def _get_video_reader(path, target_resolution):
    if target_resolution is not None:
        video = VideoReader(path, ctx=cpu(1), width=target_resolution[0], height=target_resolution[1])
    else:
        video = VideoReader(path, ctx=cpu(1))
        
    return video

def read_video_file(path, start=0, end=None, target_ordering="c t h w", target_resolution=(426, 240), crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    video = _get_video_reader(path, target_resolution)
    return read_video_object(video, start, end, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)


def read_video_object(video, start=0, end=None, target_ordering="c t h w", crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    # End is excluded
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )
    
    fps = video.get_avg_fps()
    frames_taken = range(start, min(len(video), end), nth_frames)
    try:
        video_frames = video.get_batch(frames_taken)
    except Exception as e:
        print(f"Warning: Failed read video as batch, going to frame by frame reading. The cause was: {str(e)}")
        video_frames = []
        current_frame = 0
        frames_taken = list(frames_taken)
        try:
            video.seek(start)
        except Exception as e:
            raise IllegalStartException(f"Error: Failed to seek start frame '{start}' of video. The video is likely shorter than '{start}' frames. The cause was: {str(e)}")
        for i in range(min(len(video), end)):
            try:
                if current_frame in frames_taken:
                    video_frames.append(video.next())   
                current_frame += 1
            except Exception as e:
                if(broken_frame_warning_only):
                    print(f"Warning: Failed to load frame '{i}' of video, frame will be skipped. The cause was: {str(e)}")
                else:
                    raise IllegalFrameException(f"Error: Failed to load frame '{i}' of video. The cause was: {str(e)}")
        
        video_frames = torch.stack(video_frames)
        
    if crop_frames is not None:
        video_frames = video_frames.permute(0, 3, 2, 1)
        transform = transforms.CenterCrop(crop_frames)
        video_frames = transform(video_frames)
        video_frames = video_frames.permute(0, 3, 2, 1)
    
    if len(video_frames.shape) == 3:
        video_frames = video_frames.unsqueeze(0)

    video_frames = rearrange(video_frames, f"t h w c -> {target_ordering}")
    return video_frames, fps


def read_video_file_as_parts(path, frames_per_part, target_ordering="c t h w", target_resolution=(426, 240), crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    video = _get_video_reader(path, target_resolution)
    return read_video_object_as_parts(video, frames_per_part, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)


def read_video_object_as_parts(video, frames_per_part, target_ordering="c t h w", crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    start = 0
    end = frames_per_part
    frames = []
    frame_fps = []
    
    while(True):
        frame, fps = read_video_object(video, start, end, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)
        frames.append(frame)  
        frame_fps.append(fps)           
        
        if frames_per_part is None or frame.shape[0] < frames_per_part:
            break
            
        start = end
        end += frames_per_part
    
    return frames, frame_fps


def write_video_object(path, torch_data, video_ordering="c t h w", target_resolution=None, fps=25, writing_codec='mp4v'):
    torch_data = rearrange(torch_data, f"{video_ordering} -> t h w c")
    
    if target_resolution is None:
        target_resolution = (torch_data.shape[2], torch_data.shape[1])
    
    fourcc = cv2.VideoWriter_fourcc(*writing_codec)
    out = cv2.VideoWriter(path, fourcc, fps, target_resolution)
    for i in range(torch_data.shape[0]):
        frame = cv2.cvtColor(numpy.uint8(torch_data[i].numpy()), cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()

