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

# Set the decord backend to torch
# This is hardcoded because we never used
# a different backend in the project
decord.bridge.set_bridge('torch')

class IllegalStartException(Exception):
    """
    Raised when the start frame is larger than the video length.
    """    
    pass

class IllegalFrameException(Exception):
    """
    Raised when a frame cannot be read.
    """    
    pass

def _get_video_reader(path, target_resolution):
    """
    Get a video reader for the given path and target resolution.

    :param path: Path to the video file.
    :param target_resolution: The target resolution of the video.
    :return: A video reader for the given path and target resolution.
    """    
    if target_resolution is not None:
        video = VideoReader(path, ctx=cpu(1), width=target_resolution[0], height=target_resolution[1])
    else:
        video = VideoReader(path, ctx=cpu(1))
        
    return video

def read_video_file(path, start=0, end=None, target_ordering="c t h w", target_resolution=(426, 240), crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    """
    Read a video file as a tuple containing the video frames and the fps of the video.

    :param path: Path to the video file.
    :param start: Start frame, defaults to 0
    :param end: End frame, defaults to None which means the end of the video.
    :param target_ordering: The order of the dimensions of the returned tensor, defaults to "c t h w"
    :param target_resolution: The target resolution of the video, defaults to (426, 240)
    :param crop_frames: Crop the frames to the given size, defaults to None
    :param nth_frames: Only read every nth frame, defaults to 1
    :param broken_frame_warning_only: If true, only print a warning when a frame cannot be read, defaults to True
    :return: A tuple containing the video frames and the fps of the video.
    """    
    video = _get_video_reader(path, target_resolution)
    return read_video_object(video, start, end, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)


def read_video_object(video, start=0, end=None, target_ordering="c t h w", crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    """
    Read a video object as a tuple containing the video frames and the fps of the video.

    :param video: The video object.
    :param start: Start frame, defaults to 0
    :param end: End frame, defaults to None which means the end of the video.
    :param target_ordering: The order of the dimensions of the returned tensor, defaults to "c t h w"
    :param crop_frames: Crop the frames to the given size, defaults to None
    :param nth_frames: Only read every nth frame, defaults to 1
    :param broken_frame_warning_only: If true, only print a warning when a frame cannot be read, defaults to True
    :raises ValueError: If the end frame is smaller than the start frame.
    :raises IllegalStartException: If the start frame is larger than the video length.
    :raises IllegalFrameException: If a frame cannot be read.
    :return: A tuple containing the video frames and the fps of the video.
    """    
    # End is excluded
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )
    
    # Get the video fps
    fps = video.get_avg_fps()
    # Get the frames to take
    frames_taken = range(start, min(len(video), end), nth_frames)
    # Try to read the video as a batch
    try:
        video_frames = video.get_batch(frames_taken)
    except Exception as e:
        # If it fails, read frame by frame
        print(f"Warning: Failed read video as batch, going to frame by frame reading. The cause was: {str(e)}")
        video_frames = []
        current_frame = 0
        frames_taken = list(frames_taken)
        try:
            # Seek to the start frame
            video.seek(start)
        except Exception as e:
            raise IllegalStartException(f"Error: Failed to seek start frame '{start}' of video. The video is likely shorter than '{start}' frames. The cause was: {str(e)}")
        # Try to read the frames, frame by frame
        for i in range(min(len(video), end)):
            # Read the next frame
            try:
                if current_frame in frames_taken:
                    video_frames.append(video.next())   
                current_frame += 1
            except Exception as e:
                if(broken_frame_warning_only):
                    print(f"Warning: Failed to load frame '{i}' of video, frame will be skipped. The cause was: {str(e)}")
                else:
                    raise IllegalFrameException(f"Error: Failed to load frame '{i}' of video. The cause was: {str(e)}")
        
        # Convert the frames to a tensor
        video_frames = torch.stack(video_frames)
    
    # Crop the frames if needed
    if crop_frames is not None:
        video_frames = video_frames.permute(0, 3, 2, 1)
        transform = transforms.CenterCrop(crop_frames)
        video_frames = transform(video_frames)
        video_frames = video_frames.permute(0, 3, 2, 1)
    
    if len(video_frames.shape) == 3:
        video_frames = video_frames.unsqueeze(0)

    # Reorder the dimensions
    video_frames = rearrange(video_frames, f"t h w c -> {target_ordering}")
    return video_frames, fps


def read_video_file_as_parts(path, frames_per_part, target_ordering="c t h w", target_resolution=(426, 240), crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    """
    Read a video file as a list of tuples containing multiple part of video frames and the fps of each part.

    :param path: Path to the video file.
    :param frames_per_part: The number of frames per part.
    :param target_ordering: The order of the dimensions of the returned tensor, defaults to "c t h w"
    :param target_resolution: The target resolution of the video, defaults to (426, 240)
    :param crop_frames: Crop the frames to the given size, defaults to None
    :param nth_frames: Only read every nth frame, defaults to 1
    :param broken_frame_warning_only: If true, only print a warning when a frame cannot be read, defaults to True
    :return: A list of tuples containing multiple part of video frames and the fps of each part.
    """    
    video = _get_video_reader(path, target_resolution)
    return read_video_object_as_parts(video, frames_per_part, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)


def read_video_object_as_parts(video, frames_per_part, target_ordering="c t h w", crop_frames=None, nth_frames = 1, broken_frame_warning_only=True):
    """
    Read a video object as a list of tuples containing multiple part of video frames and the fps of each part.

    :param video: The video object.
    :param frames_per_part: The number of frames per part.
    :param target_ordering: The order of the dimensions of the returned tensor, defaults to "c t h w"
    :param crop_frames: Crop the frames to the given size, defaults to None
    :param nth_frames: Only read every nth frame, defaults to 1
    :param broken_frame_warning_only: If true, only print a warning when a frame cannot be read, defaults to True
    :return: A list of tuples containing multiple part of video frames and the fps of each part.
    """    
    start = 0
    end = frames_per_part
    frames = []
    frame_fps = []
    
    # Read the video in parts, as long as the video part is longer than the frames per part
    while(True):
        frame, fps = read_video_object(video, start, end, target_ordering, crop_frames, nth_frames, broken_frame_warning_only)
        frames.append(frame)  
        frame_fps.append(fps)           
        
        # Break if the video is shorter than the frames per part
        if frames_per_part is None or frame.shape[0] < frames_per_part:
            break
            
        start = end
        end += frames_per_part
    
    return frames, frame_fps


def write_video_object(path, torch_data, video_ordering="c t h w", target_resolution=None, fps=25, writing_codec='mp4v'):
    """
    Write a video object to a video file using OpenCV.

    :param path: Path to the video file.
    :param torch_data: The video object as a tensor.
    :param video_ordering: The order of the dimensions of the tensor, defaults to "c t h w"
    :param target_resolution: The target resolution of the written video, defaults to None which means the resolution of the tensor.
    :param fps: The fps of the written video, defaults to 25
    :param writing_codec: The codec used to write the video, defaults to 'mp4v'
    """    
    torch_data = rearrange(torch_data, f"{video_ordering} -> t h w c")
    
    if target_resolution is None:
        target_resolution = (torch_data.shape[2], torch_data.shape[1])
    
    fourcc = cv2.VideoWriter_fourcc(*writing_codec)
    out = cv2.VideoWriter(path, fourcc, fps, target_resolution)
    for i in range(torch_data.shape[0]):
        frame = cv2.cvtColor(numpy.uint8(torch_data[i].numpy()), cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()

