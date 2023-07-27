from WebvidReader.VideoDataset import VideoDataset
import numpy
import time
from tqdm import tqdm
import cv2

csv_path = "/home/shared-data/webvid/results_10M_val.csv"
data_path = "/home/shared-data/webvid/data_val/videos"

dataset_non_pickle = VideoDataset(
    csv_path, 
    data_path, 
    target_resolution=(64, 64), 
    target_ordering="t c h w",
    max_frames_per_part=16,
    nth_frames=5,
    first_part_only=True,
    min_frames_per_part=0
)

print("Profiling Non Pickle")
start = time.perf_counter()
max_ind = 100
for i in tqdm(range(max_ind)):
    video, caption, fps = dataset_non_pickle[i]
print("Non Pickle took: " + str(time.perf_counter() - start) + " seconds!")

dataset = VideoDataset(
    csv_path, 
    data_path, 
    target_resolution=(64, 64), 
    target_ordering="t c h w",
    max_frames_per_part=16,
    nth_frames=5,
    first_part_only=True,
    min_frames_per_part=0,
    pickle_vid_data=True
)
print("Preparing Pickle")
for i in tqdm(range(100)):
    video, caption, fps = dataset[i]

    
print("Profiling Pickle")

start = time.perf_counter()
for i in tqdm(range(100)):
    video, caption, fps = dataset[i]
print("Pickle took: " + str(time.perf_counter() - start) + " seconds!")

out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (dataset[1][0].shape[2],dataset[1][0].shape[1]))

for frame in dataset[1][0]:
    out.write(numpy.uint8(frame.numpy()))
    
out.release()

