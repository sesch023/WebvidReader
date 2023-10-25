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

# A named tuple to store the path and the caption of a video.
VideoItem = namedtuple("VideoItem", ["Caption", "Path"])

class VideoDataset(Dataset):
    @staticmethod
    def parse_csv(csv_path, verbose=False):
        """
        Parses the csv file and returns a dictionary of VideoItems and a list of keys.
        The csv file is in the format given by the WebVid-10M dataset.

        :param csv_path: Path to the csv file.
        :param verbose: If true, a progress bar will be shown.
        :return: A tuple of a dictionary of VideoItems and a list of keys with video ids.
        """        
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
        """
        Creates a new VideoDataset.

        :param csv_path: Path to the csv file.
        :param video_base_path: Path to the directory containing the videos.
        :param target_ordering: The ordering of the video dimensions. The default is "c t h w", 
                                which means that the video has the dimensions (channels, time, height, width).
        :param target_resolution: The target resolution of the video. The default is (426, 240).
        :param crop_frames: If not None, the video will be cropped to the given resolution.
        :param pickle_vid_data: If true, the video data will be pickled to the given pickle_base_path. 
                                This can be used to speed up the loading of the dataset, but it requires
                                huge amounts of disk space.
        :param pickle_base_path: The base path for the pickled video data.
        :param verbose: If true, a progress bar will be shown when parsing the csv file.
        :param first_part_only: If true, only the first part of the video defined by max and min frames per part will be loaded, 
                                all frames after the first part will be discarded.
        :param max_frames_per_part: The maximum number of frames per part.
        :param min_frames_per_part: The minimum number of frames per part.
        :param nth_frames: Only every nth frame will be loaded.
        :param normalize: If true, the video data will be normalized to the range [-1, 1].
        """        
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
        
        self._video_map, self._keys = self.parse_csv(csv_path, verbose=verbose)
        self._repickle = False
        
        # Check if the pickle file exists and check if the dataset has changed.
        if pickle_vid_data:
            dataset_pickle = f"{pickle_base_path}/dataset.bin"
            if os.path.isfile(dataset_pickle):
                # Load the old dataset.
                with open(dataset_pickle, "rb") as f:
                    old = pickle.load(f)
                # Check if the dataset has changed. If yes set the repickle flag.
                if not(self.__old_equals_self__(old)):
                    self._repickle = True
            with open(dataset_pickle, "wb") as f:
                pickle.dump(self, f)
    
    def __old_equals_self__(self, old):
        """
        Checks if all the relevant values in the given VideoDataset are equal to the value in this VideoDataset.

        :param old: The VideoDataset to compare to.
        :return: True if all relevant values are equal, False otherwise.
        """        
        return (self._target_ordering == old._target_ordering and self._target_resolution == old._target_resolution and self._csv_path == old._csv_path and self._video_base_path == old._video_base_path and self._max_frames_per_part == old._max_frames_per_part and self._first_part_only == old._first_part_only and self._min_frames_per_part == old._min_frames_per_part and self._normalize == old._normalize)

    def normalize(self, data):
        """
        Normalizes the given data from [0, 255] to the range [-1, 1].
        Since the package was so far only used for diffusion, this was hardcoded.

        :param data: The data to normalize.
        :return: The normalized data.
        """        
        return VideoDataset.normalize(data)
    
    def reverse_normalize(self, data):
        """
        Reverses the normalization of the given data from the range [-1, 1] to [0, 255].
        Since the package was so far only used for diffusion, this was hardcoded.

        :param data: The data to reverse normalize.
        :return: The reverse normalized data.
        """        
        return VideoDataset.reverse_normalize(data)
    
    @staticmethod
    def normalize(data):
        """
        Normalizes the given data from [0, 255] to the range [-1, 1].
        Since the package was so far only used for diffusion, this was hardcoded.

        :param data: The data to normalize.
        :return: The normalized data.
        """        
        return torch.add(torch.div(data, 255/2), -1) 
    
    @staticmethod
    def reverse_normalize(data):
        """
        Reverses the normalization of the given data from the range [-1, 1] to [0, 255].

        :param data: The data to reverse normalize.
        :return: The reverse normalized data.
        """        
        return torch.mul(torch.add(data, 1), 255/2)
    
    def __len__(self):
        """
        Returns the number of videos in the dataset.

        :return: The number of videos in the dataset.
        """        
        return len(self._video_map)

    def __getitem__(self, idx):
        """
        Returns the video at the given index.

        :param idx: The index of the video to return.
        :return: A tuple of the video, the label and the fps of the video.
        """        
        key = self._keys[idx]
        video_meta = self._video_map[key]
        # Check if the video is already pickled and if it should be loaded from the pickle.
        pickle_path = f"{self._pickle_base_path}/{video_meta.Path.replace('/', '')}.nbz" if self._pickle_vid_data else None
        # Check if the pickle file exists and if it should be loaded.
        load_pickle = self._pickle_vid_data and os.path.isfile(pickle_path) and not self._repickle
        fps = None
        
        # Try to load the video from the pickle file.
        if load_pickle:
            try:
                with open(pickle_path, "rb") as f:
                    video, fps = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load numpy pickle '{pickle_path}', falling back to reading the according video file. The cause was: {str(e)}")
                load_pickle = False
        
        # If the video was not loaded from the pickle file, load it from the video file.
        if not load_pickle:
            vid_path = f"{self._video_base_path}/{video_meta.Path}"
            
            try:
                # If only the first part of the video should be loaded, load the video as a whole with the given range.
                if self._first_part_only:
                    video, fps = read_video_file(vid_path, target_ordering=self._target_ordering, target_resolution=self._target_resolution, end=self._max_frames_per_part, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                    video = None if video.shape[self._time_dim] < self._min_frames_per_part else video
                # If the whole video should be loaded, load the video as parts with the given range.
                else:
                    video, fps = read_video_file_as_parts(vid_path, self._max_frames_per_part, target_ordering=self._target_ordering, target_resolution=self._target_resolution, crop_frames=self._crop_frames, nth_frames=self._nth_frames)
                    video = list(filter(lambda el: el.shape[self._time_dim] >= self._min_frames_per_part, video)) if self._min_frames_per_part > 0 else video
                
                # If the video should be pickled, pickle it.
                if self._pickle_vid_data:
                    with open(pickle_path, "wb") as f:
                        pickle.dump((video, fps), f)
            except Exception as e:
                print(f"Warning: Failed to load MP4 '{vid_path}', the Data will be returned as None. The cause was: {str(e)}")
                video = None
        
        # If the video is not None, convert it to a tensor and normalize it.
        label = video_meta.Caption
        if video is not None:
            # If only the first part of the video should be loaded, convert the video to a tensor.
            if self._first_part_only:
                video = torch.Tensor(video).float()
                if self._normalize:
                    video = self.normalize(video)
            # If the whole video should be loaded, convert the video parts to tensors.
            else:
                for i in range(len(video)):
                    video[i] = torch.Tensor(video[i]).float() if video[i] is not None else None
                    if video[i] is not None and self._normalize:
                        video[i] = self.normalize(video[i])
            
        return video, label, fps
    
    def write_item(self, idx, path):
        """
        Writes the video at the given index to the given path.

        :param idx: The index of the video to write.
        :param path: The path to write the video to.
        """        
        video, label = self.__getitem__(idx)
        self.write_video(video, path)

    def write_video(self, video, path):
        """
        Writes the given video to the given path.

        :param video: The video to write.
        :param path: The path to write the video to.
        """        
        if video is None or (isinstance(video, list) and any(map(lambda x: x is None, video))):
            print("Warning: Cannot write video with missing parts.")
        
        target_resolution= self._crop_frames if self._crop_frames is not None else self._target_resolution
        
        # If the video is not a list, it is a single video.
        if not isinstance(video, list):
            if self._normalize:
                video = self.reverse_normalize(video)
            
            write_video_object(path, video, video_ordering=self._target_ordering, target_resolution=target_resolution)
        # If the video is a list, it is a list of parts.
        else:
            path_f = path[:-4] + "_{num}" + path[-4:]
            for i in range(len(video)):
                if video[i] is not None and self._normalize:
                    video[i] = self.reverse_normalize(video[i])
                path_c = path_f.format(num=i)
                write_video_object(path_c, video[i], video_ordering=self._target_ordering, target_resolution=target_resolution)
        

