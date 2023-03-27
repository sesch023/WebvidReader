import torch
import pandas
from torch.utils.data import Dataset
from collections import namedtuple
from Video import read_video_file

VideoItem = namedtuple("VideoItem", ["Id", "Page", "Caption", "Path"])


class VideoDataset(Dataset):
    @staticmethod
    def parse_csv(csv_path, video_base_path):
        items = dict()
        csv = pandas.read_csv(csv_path)
        keys = list(csv["videoid"])
        for index, row in csv.iterrows():
            path = f"{video_base_path}/{row['page_dir']}/{row['videoid']}.mp4"
            item = VideoItem(Id=row['videoid'], Page=row['page_dir'], Caption=row['name'], Path=path)
            items[row['videoid']] = item

        return items, keys

    def __init__(self, csv_path, video_base_path):
        self._csv_path = csv_path
        self._video_base_path = video_base_path
        self._video_map, self._keys = self.parse_csv(csv_path, video_base_path)

    def __len__(self):
        return len(self._video_map)

    def __getitem__(self, idx):
        key = self._keys[idx]
        print((self._video_map[key]).Path)
        video = read_video_file((self._video_map[key]).Path)
        label = self._video_map[key].Caption
        return video, label

