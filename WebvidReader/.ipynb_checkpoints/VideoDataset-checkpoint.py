import pandas
import torch
import numpy
from torch.utils.data import Dataset
from collections import namedtuple
from WebvidReader.Video import read_video_file
import pickle
import os
from tqdm import tqdm

VideoItem = namedtuple("VideoItem", ["Id", "Page", "Caption", "Path", "Pickle"])


class VideoDataset(Dataset):
    @staticmethod
    def __get_pickle_path__(pickle_vid_data, pickle_path, video_id):
        return f"{pickle_path}/{video_id}.npz" if pickle_vid_data else None
    
    @staticmethod
    def parse_csv(csv_path, video_base_path, pickle_vid_data=False, pickle_base_path=".video_pickles", verbose=False):
        items = dict()
        csv = pandas.read_csv(csv_path)
        keys = list(csv["videoid"])
        
        if verbose:
            print("Creating VideoDataset")
        
        itererator = tqdm(csv.iterrows(), total=csv.shape[0]) if verbose else csv.iterrows()
        for index, row in itererator:
            path = f"{video_base_path}/{row['page_dir']}/{row['videoid']}.mp4"
            pickle_path = VideoDataset.__get_pickle_path__(pickle_vid_data, pickle_base_path, row['videoid'])
            item = VideoItem(Id=row['videoid'], Page=row['page_dir'], Caption=row['name'], Path=path, Pickle=pickle_path)
            items[row['videoid']] = item

        return items, keys

    def __init__(self, csv_path, video_base_path, channels_first=False, target_resolution=(426, 240), pickle_vid_data=False, pickle_base_path="video_pickles", verbose=True):
        self._csv_path = csv_path
        self._video_base_path = video_base_path
        self._channels_first = channels_first
        self._target_resolution = target_resolution
        self._pickle_vid_data = pickle_vid_data
        
        if pickle_vid_data and not os.path.exists(pickle_base_path):
            os.makedirs(pickle_base_path)
        
        self._video_map, self._keys = self.parse_csv(csv_path, video_base_path, pickle_vid_data, pickle_base_path, verbose=verbose)
        self._repickle = False
        
        if pickle_vid_data:
            dataset_pickle = f"{pickle_base_path}/dataset.bin"
            if os.path.isfile(dataset_pickle):
                with open(dataset_pickle, "rb") as f:
                    old = pickle.load(f)
                if not(channels_first == old._channels_first and target_resolution == old._target_resolution and csv_path == old._csv_path and video_base_path == old._video_base_path):
                    self._repickle = True
            with open(dataset_pickle, "wb") as f:
                pickle.dump(self, f)
        

    def __len__(self):
        return len(self._video_map)

    def __getitem__(self, idx):
        key = self._keys[idx]
        video_meta = self._video_map[key]
        load_pickle = self._pickle_vid_data and os.path.isfile(video_meta.Pickle) and not self._repickle
        
        if load_pickle:
            try:
                video = numpy.load(video_meta.Pickle, allow_pickle=False)
            except Exception as e:
                print(f"Warning: Failed to load numpy pickle '{video_meta.Pickle}', falling back to reading the according video file. The cause was: {str(e)}")
                load_pickle = False
        
        if not load_pickle:
            video = read_video_file(video_meta.Path, channels_first=self._channels_first, target_resolution=self._target_resolution)
            if self._pickle_vid_data:
                with open(video_meta.Pickle, "wb") as f:
                    numpy.save(f, video, allow_pickle=False)
                    
        label = video_meta.Caption
        return torch.Tensor(video), label

