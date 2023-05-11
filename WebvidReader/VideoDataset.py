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

    def __init__(self, csv_path, video_base_path, channels_first=False, target_resolution=(426, 240), crop_frames=None, pickle_vid_data=False, pickle_base_path="video_pickles", verbose=True, max_frames_per_part=None, first_frame_only=True, nth_frames=1):
        self._csv_path = csv_path
        self._video_base_path = video_base_path
        self._channels_first = channels_first
        self._target_resolution = target_resolution
        self._pickle_vid_data = pickle_vid_data
        self._video_base_path = video_base_path
        self._pickle_base_path = pickle_base_path
        self._max_frames_per_part = max_frames_per_part
        self._crop_frames = crop_frames
        self._nth_frames = nth_frames
        self._first_frame_only = first_frame_only
        
        if pickle_vid_data and not os.path.exists(pickle_base_path):
            os.makedirs(pickle_base_path)
        
        self._video_map, self._keys = self.parse_csv(csv_path, pickle_vid_data, pickle_base_path, verbose=verbose)
        self._repickle = False
        
        if pickle_vid_data:
            dataset_pickle = f"{pickle_base_path}/dataset.bin"
            if os.path.isfile(dataset_pickle):
                with open(dataset_pickle, "rb") as f:
                    old = pickle.load(f)
                if not(channels_first == old._channels_first and target_resolution == old._target_resolution and csv_path == old._csv_path and video_base_path == old._video_base_path and max_frames_per_part == old._max_frames_per_part and first_frame_only == old._first_frame_only):
                    self._repickle = True
            with open(dataset_pickle, "wb") as f:
                pickle.dump(self, f)
        

    def __len__(self):
        return len(self._video_map)

    def __getitem__(self, idx):
        key = self._keys[idx]
        video_meta = self._video_map[key]
        pickle_path = f"{self._pickle_base_path}/{video_meta.Path.replace('/', '')}.nbz" if self._pickle_vid_data else None
        load_pickle = self._pickle_vid_data and os.path.isfile(pickle_path) and not self._repickle
        
        if load_pickle:
            try:
                with open(pickle_path, "rb") as f:
                    video = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load numpy pickle '{pickle_path}', falling back to reading the according video file. The cause was: {str(e)}")
                load_pickle = False
        
        if not load_pickle:
            vid_path = f"{self._video_base_path}/{video_meta.Path}"
            
            try:
                if self._first_frame_only:
                    video = read_video_file(vid_path, channels_first=self._channels_first, target_resolution=self._target_resolution, end=self._max_frames_per_part, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                else:
                    video = read_video_file_as_parts(vid_path, self._max_frames_per_part, channels_first=self._channels_first, target_resolution=self._target_resolution, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                    
                if self._pickle_vid_data:
                    with open(pickle_path, "wb") as f:
                        pickle.dump(video, f)
            except Exception as e:
                print(f"Warning: Failed to load MP4 '{vid_path}', the Data will be returned as None. The cause was: {str(e)}")
                video = None
                    
        label = video_meta.Caption
        
        if video is not None:
            if self._first_frame_only:
                video = torch.Tensor(video).float()
            else:
                for i in range(len(video)):
                    video[i] = torch.Tensor(video[i]).float() if video[i] is not None else None
            
        return video, label
    
    
    def write_item(self, idx, path):
        video, label = self.__getitem__(idx)
        target_resolution= self._crop_frames if self._crop_frames is not None else self._target_resolution
        
        if not isinstance(video, list):
            write_video_object(path, video, channels_first=self._channels_first, target_resolution=target_resolution)
        else:
            path_f = path[:-4] + "_{num}" + path[-4:]
            for i in range(len(video)):
                path_c = path_f.format(num=i)
                write_video_object(path_c, video[i], channels_first=self._channels_first, target_resolution=target_resolution)
        

