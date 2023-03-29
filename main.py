from WebvidReader.VideoDataset import VideoDataset

csv_path = "/qnap/homes/sesch023/Masterarbeit/webvid/results_2M_train.csv"
data_path = "/qnap/homes/sesch023/Masterarbeit/webvid/data/videos"

dataset = VideoDataset(csv_path, data_path)
print(dataset[1])


