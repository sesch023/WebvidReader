import pandas
import torch
import numpy
from torch.utils.data import Dataset
from collections import namedtuple
from WebvidReader.Video import read_video_file, write_video_object, read_video_file_as_parts
import pickle
import os
from tqdm import tqdm
import time
from einops import rearrange

VideoItem = namedtuple("VideoItem", ["Caption", "Path"])

class VideoDataset(Dataset):
    
    @staticmethod
    def parse_csv(csv_path, pickle_vid_data=False, pickle_base_path=".video_pickles", verbose=False):
        items = dict()
        csv = pandas.read_csv(csv_path)
        keys = list(csv["videoid"])
        
        if verbose:
            print("Creating VideoDataset")
        
        itererator = tqdm(csv.iterrows(), total=csv.shape[0]) if verbose else csv.iterrows()
        for index, row in itererator:
            path = f"{row['page_dir']}/{row['videoid']}.mp4"
            item = VideoItem(Caption=row['name'], Path=path)
            items[row['videoid']] = item

        return items, keys

    def __init__(self, csv_path, video_base_path, target_ordering="c t h w", target_resolution=(426, 240), crop_frames=None, pickle_vid_data=False, pickle_base_path="video_pickles", verbose=True, first_part_only=False, max_frames_per_part=float("inf"), min_frames_per_part=0, nth_frames=1, normalize=True):
        self._csv_path = csv_path
        self._video_base_path = video_base_path
        self._target_ordering = target_ordering
        self._target_resolution = target_resolution
        self._pickle_vid_data = pickle_vid_data
        self._video_base_path = video_base_path
        self._pickle_base_path = pickle_base_path
        self._max_frames_per_part = max_frames_per_part
        self._min_frames_per_part = min_frames_per_part
        self._crop_frames = crop_frames
        self._nth_frames = nth_frames
        self._first_part_only = first_part_only
        self._normalize = normalize
        self._time_dim = target_ordering.replace(" ", "").index("t")
        
        if pickle_vid_data and not os.path.exists(pickle_base_path):
            os.makedirs(pickle_base_path)
        
        self._video_map, self._keys = self.parse_csv(csv_path, pickle_vid_data, pickle_base_path, verbose=verbose)
        self._repickle = False
        
        if pickle_vid_data:
            dataset_pickle = f"{pickle_base_path}/dataset.bin"
            if os.path.isfile(dataset_pickle):
                with open(dataset_pickle, "rb") as f:
                    old = pickle.load(f)
                if not(self.__old_equals_self__(old)):
                    self._repickle = True
            with open(dataset_pickle, "wb") as f:
                pickle.dump(self, f)
    
    def __old_equals_self__(self, old):
        return (self._target_ordering == old._target_ordering and self._target_resolution == old._target_resolution and self._csv_path == old._csv_path and self._video_base_path == old._video_base_path and self._max_frames_per_part == old._max_frames_per_part and self._first_part_only == old._first_part_only and self._min_frames_per_part == old._min_frames_per_part and self._normalize == old._normalize)

    def normalize(self, data):
        return torch.add(torch.div(data, 255/2), -1) 
    
    def reverse_normalize(self, data):
        return torch.mul(torch.add(data, 1), 255/2)
    
    def __len__(self):
        return len(self._video_map)

    def __getitem__(self, idx):
        key = self._keys[idx]
        video_meta = self._video_map[key]
        pickle_path = f"{self._pickle_base_path}/{video_meta.Path.replace('/', '')}.nbz" if self._pickle_vid_data else None
        load_pickle = self._pickle_vid_data and os.path.isfile(pickle_path) and not self._repickle
        fps = None
        
        if load_pickle:
            try:
                with open(pickle_path, "rb") as f:
                    video, fps = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load numpy pickle '{pickle_path}', falling back to reading the according video file. The cause was: {str(e)}")
                load_pickle = False
        
        if not load_pickle:
            vid_path = f"{self._video_base_path}/{video_meta.Path}"
            
            try:
                if self._first_part_only:
                    video, fps = read_video_file(vid_path, target_ordering=self._target_ordering, target_resolution=self._target_resolution, end=self._max_frames_per_part, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                    video = None if video.shape[self._time_dim] < self._min_frames_per_part else video
                else:
                    video, fps = read_video_file_as_parts(vid_path, self._max_frames_per_part, target_ordering=self._target_ordering, target_resolution=self._target_resolution, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                    video = list(filter(lambda el: el.shape[self._time_dim] >= self._min_frames_per_part, video)) if self._min_frames_per_part > 0 else video
                
                
                if self._pickle_vid_data:
                    with open(pickle_path, "wb") as f:
                        pickle.dump((video, fps), f)
            except Exception as e:
                print(f"Warning: Failed to load MP4 '{vid_path}', the Data will be returned as None. The cause was: {str(e)}")
                video = None
        
        label = video_meta.Caption
        if video is not None:
            if self._first_part_only:
                video = torch.Tensor(video).float()
                if self._normalize:
                    video = self.normalize(video)
            else:
                for i in range(len(video)):
                    video[i] = torch.Tensor(video[i]).float() if video[i] is not None else None
                    if video[i] is not None and self._normalize:
                        video[i] = self.normalize(video[i])
            
        return video, label, fps
    
    def write_item(self, idx, path):
        video, label = self.__getitem__(idx)
        self.write_video(video, path)

    def write_video(self, video, path):
        if video is None or (isinstance(video, list) and any(map(lambda x: x is None, video))):
            print("Warning: Cannot write video with missing parts.")
        
        target_resolution= self._crop_frames if self._crop_frames is not None else self._target_resolution
        
        if not isinstance(video, list):
            if self._normalize:
                video = self.reverse_normalize(video)
            
            write_video_object(path, video, video_ordering=self._target_ordering, target_resolution=target_resolution)
        else:
            path_f = path[:-4] + "_{num}" + path[-4:]
            for i in range(len(video)):
                if video[i] is not None and self._normalize:
                    video[i] = self.reverse_normalize(video[i])
                path_c = path_f.format(num=i)
                write_video_object(path_c, video[i], video_ordering=self._target_ordering, target_resolution=target_resolution)
        

