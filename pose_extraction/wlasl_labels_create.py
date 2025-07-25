import utils as utils
import os
import pickle
import gzip

split="dev"
path="/home/zeynep/Thesis/code/Uni-Sign/data/WLASL/labels-2000."+split
raw_data = utils.load_dataset_file(path)

wlasl100="/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/rgb_format/"+split+".txt"

target_keys =[]
with open(wlasl100, "r") as f:  # dosya adını buraya yaz
    for line in f:
        path_part = line.strip().split()[0]      # örn. 'train/65225.mp4'
        filename = os.path.splitext(os.path.basename(path_part))[0]
        target_keys.append(filename)


subset_data = {k: raw_data[k] for k in target_keys if k in raw_data}

# Dosyaya gzip + pickle ile kaydet
with gzip.open("/home/zeynep/Thesis/code/Uni-Sign/data/WLASL100/labels-100."+split, "wb") as f:
    pickle.dump(subset_data, f)