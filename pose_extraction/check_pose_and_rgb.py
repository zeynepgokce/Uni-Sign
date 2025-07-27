

import os
import pickle
from decord import VideoReader, cpu
import numpy as np
import random

def load_pose(pose_dir, path):
	pose = pickle.load(open(os.path.join(pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
	duration = len(pose['scores'])
	if duration > 16:
		tmp = sorted(random.sample(range(duration), k=16))
	else:
		tmp = list(range(duration))

	tmp = np.array(tmp)
	#print(tmp)

	skeletons = pose['keypoints']
	confs = pose['scores']
	skeletons_tmp = []
	confs_tmp = []
	for index in tmp:
		skeletons_tmp.append(skeletons[index])
		confs_tmp.append(confs[index])

	skeletons = skeletons_tmp
	confs = confs_tmp
	confs = np.array(confs)
	print("confs: ", confs.shape)
	for i in range(len(confs)):
		if (confs[i].shape !=(1,133)):
			print("************************SHAPE*********************************")

	left_confs_filter = confs[:, 0, 91:112].mean(-1)
	right_confs_filter = confs[:, 0, 112:].mean(-1)


	return duration


def load_video_support_rgb(path):
	path= path.replace(".pkl", '.mp4')
	vr = VideoReader(path, num_threads=1, ctx=cpu(0))
	return len(vr)



pose_dir = "/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/pose_format"
rgb_dir = "/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/rgb_format"
split= "train"



folder_path = os.path.join(pose_dir, split)
all_files = [ f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for f in all_files:
	print(f)
	pose_len = load_pose(pose_dir+"/"+split, f)
	video_len = load_video_support_rgb(os.path.join(rgb_dir+"/"+split, f))
	if pose_len != video_len:
		print("length mismacthing")
		print("video len: ",video_len)
		print("pose len: ", pose_len)