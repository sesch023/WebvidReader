from WebvidReader.VideoDataset import VideoDataset
import cv2
import numpy
import time
from tqdm import tqdm

csv_path = "/home/shared-data/webvid/results_10M_val.csv"
data_path = "/home/shared-data/webvid/data_val/videos"

dataset_non_pickle = VideoDataset(csv_path, data_path, channels_first=False, pickle_vid_data=False, target_resolution=(426, 240))
print("Profiling Non Pickle")
start = time.perf_counter()
for i in tqdm(range(100)):
    video = dataset_non_pickle[i]
print("Non Pickle took: " + str(time.perf_counter() - start) + " seconds!")

dataset = VideoDataset(csv_path, data_path, channels_first=False, pickle_vid_data=True, target_resolution=(426, 240))

print("Preparing Pickle")
for i in tqdm(range(100)):
    video = dataset[i]

    
print("Profiling Pickle")

start = time.perf_counter()
for i in tqdm(range(100)):
    video = dataset[i]
print("Pickle took: " + str(time.perf_counter() - start) + " seconds!")

out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (dataset[1][0].shape[2],dataset[1][0].shape[1]))

for frame in dataset[1][0]:
    out.write(numpy.uint8(frame.numpy()))
    
out.release()
cv2.destroyAllWindows()


