import argparse
import os
import cv2
import glob
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody, draw_skeleton
bad_videos = []
def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape

    # En güvenilir kişiyi seç: skor ortalaması en yüksek olan
    if len(scores.shape) == 2 and scores.shape[0] > 1:
        mean_scores = scores.mean(axis=1)  # kişi bazında ortalama
        best_idx = np.argmax(mean_scores)
        keypoints = keypoints[best_idx]  # (133, 2)
        scores = scores[best_idx]  # (133,)
    else:
        keypoints = keypoints[0]  # (133, 2)
        scores = scores[0]  # (133,)

    print("keypoint shape: ", keypoints.shape)
    print("scores shape: ", scores.shape)
    return keypoints, scores, [W, H]

def process_video(video_path, tgt_dir, wholebody, max_workers=16, overwrite=False):
    output_path = os.path.join(tgt_dir, os.path.basename(video_path).replace(".mp4", ".pkl"))
    if os.path.exists(output_path) and not overwrite:
        print(f"file already exists, skip: {output_path}")
        return

    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return

    vid_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vid_data.append(frame)
    cap.release()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in vid_data]
        for f in tqdm(futures, desc="Processing frames", total=len(vid_data)):
            results.append(f.result())

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

        if scores.shape != (1,133):
            print(video_path)
            bad_videos.append((video_path, scores.shape))


    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", default="/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/rgb_format/train", help="video dir path")
    parser.add_argument("--tgt_dir", default="/home/zeynep/Thesis/code/Uni-Sign/dataset/WLASL100_64x64/pose_format/train", help="pose dir path")

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])

    parser.add_argument("--video_extensions", nargs='+', default=["mp4"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    video_files = []
    for ext in args.video_extensions:
        video_files.extend(glob.glob(os.path.join(args.src_dir, f'*.{ext}')))

    print(f"found {len(video_files)} videos")

    for video_path in tqdm(video_files, desc="Processing"):
        process_video(
            video_path=video_path,
            tgt_dir=args.tgt_dir,
            wholebody=wholebody,
            max_workers=args.max_workers,
            overwrite=args.overwrite
        )
    with open("./bad_pose_files.txt", "w") as f:
        for path, shape in bad_videos:
            f.write(f"{path}\t{shape}\n")


if __name__ == "__main__":
    main()
