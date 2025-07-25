

import os
import pickle
from decord import VideoReader, cpu

def load_pose(pose_dir, path):
	pose = pickle.load(open(os.path.join(pose_dir, path.replace(".mp4", '.pkl')), 'rb'))

	if 'start' in pose.keys():
		assert pose['start'] < pose['end']
		duration = pose['end'] - pose['start']
		start = pose['start']
	else:
		duration = len(pose['scores'])
		start = 0
	return duration


def load_video_support_rgb(path):
	vr = VideoReader(path, num_threads=1, ctx=cpu(0))
	return len(vr)




pose_dir = "/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/pose_format"
rgb_dir = "/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/rgb_format"
split= "train"


folder_path = os.path.join(rgb_dir, split)
all_files = [ f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for f in all_files:
	pose_len = load_pose(pose_dir+"/"+split, f)
	video_len = load_video_support_rgb(os.path.join(rgb_dir+"/"+split, f))
	if  pose_len!=video_len:
		print(f)