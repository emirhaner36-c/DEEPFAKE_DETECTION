"""
Extracts frames from FaceForensics++ videos and builds manifests.

FaceForensics++ must be requested from the authors at:
  https://github.com/ondyari/FaceForensics

Expected layout after download:
    data/raw/ff++/
        original_sequences/actors/c23/videos/      ← real
        manipulated_sequences/Deepfakes/c23/videos/
        manipulated_sequences/Face2Face/c23/videos/
        manipulated_sequences/FaceSwap/c23/videos/
        manipulated_sequences/NeuralTextures/c23/videos/

Frames are extracted at a fixed FPS, then face-cropped.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch
import pandas as pd
from tqdm import tqdm
from src.preprocessing import build_mtcnn, detect_and_crop

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def extract_frames(video_path: str, out_dir: str, fps: int = 1, max_frames: int = 30) -> list[str]:
    """Extract up to max_frames frames at the given FPS from a video."""
    os.makedirs(out_dir, exist_ok=True)
    cap     = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval  = max(1, int(video_fps / fps))

    saved, frame_idx, paths = 0, 0, []
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            paths.append(out_path)
            saved += 1
        frame_idx += 1

    cap.release()
    return paths


def process_video_dir(
    video_dir: str,
    frame_out_dir: str,
    face_out_dir: str,
    label: int,
    mtcnn,
    fps: int = 1,
    max_frames: int = 30,
    max_videos: int = 200,
) -> list[dict]:
    records = []
    videos  = list(Path(video_dir).glob("*.mp4"))[:max_videos]

    for vid in tqdm(videos, desc=Path(video_dir).parent.name):
        vid_stem  = vid.stem
        f_out_dir = os.path.join(frame_out_dir, vid_stem)
        face_dir  = os.path.join(face_out_dir,  vid_stem)

        frame_paths = extract_frames(str(vid), f_out_dir, fps, max_frames)

        for fp in frame_paths:
            fname    = os.path.basename(fp)
            face_out = os.path.join(face_dir, fname)
            success  = detect_and_crop(fp, mtcnn, face_out)
            if success:
                records.append({"path": face_out, "label": label})

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",      default="data/raw/ff++")
    parser.add_argument("--frame_dir",    default="data/raw/ff++_frames")
    parser.add_argument("--out_dir",      default="data/processed/ff++")
    parser.add_argument("--manifest_dir", default="data/manifests")
    parser.add_argument("--fps",          type=int, default=1)
    parser.add_argument("--max_frames",   type=int, default=30)
    parser.add_argument("--max_videos",   type=int, default=200)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split",        default="0.70,0.15,0.15")
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    mtcnn = build_mtcnn(args.device)

    all_records = []

    # Real videos
    real_video_dir = os.path.join(args.raw_dir, "original_sequences/actors/c23/videos")
    if os.path.isdir(real_video_dir):
        logger.info("Processing REAL videos...")
        all_records += process_video_dir(
            real_video_dir,
            os.path.join(args.frame_dir, "real"),
            os.path.join(args.out_dir,   "real"),
            label=0, mtcnn=mtcnn,
            fps=args.fps, max_frames=args.max_frames, max_videos=args.max_videos,
        )

    # Fake videos (all manipulation methods)
    for method in FAKE_METHODS:
        video_dir = os.path.join(args.raw_dir, f"manipulated_sequences/{method}/c23/videos")
        if os.path.isdir(video_dir):
            logger.info(f"Processing FAKE ({method}) videos...")
            all_records += process_video_dir(
                video_dir,
                os.path.join(args.frame_dir, method),
                os.path.join(args.out_dir,   method),
                label=1, mtcnn=mtcnn,
                fps=args.fps, max_frames=args.max_frames, max_videos=args.max_videos,
            )

    # Build manifests
    import random
    random.shuffle(all_records)
    ratios    = [float(x) for x in args.split.split(",")]
    n         = len(all_records)
    n_train   = int(n * ratios[0])
    n_val     = int(n * ratios[1])

    splits = {
        "train": all_records[:n_train],
        "val":   all_records[n_train:n_train + n_val],
        "test":  all_records[n_train + n_val:],
    }

    for name, records in splits.items():
        df  = pd.DataFrame(records)
        out = os.path.join(args.manifest_dir, f"{name}.csv")
        df.to_csv(out, index=False)
        logger.info(f"{name}: {len(df)} frames — Real: {sum(df.label==0)} | Fake: {sum(df.label==1)}")


if __name__ == "__main__":
    main()
