
import cv2
import os
import argparse
import json
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, frames_per_second: float):
    """Extracts frames from a video file and saves them as PNGs.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save the extracted frames.
        frames_per_second: Number of frames to extract per second of video.
    """
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"\nProcessing video: {video_path.name}")
    print(f"  - FPS: {video_fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Extracting {frames_per_second} frame(s) per second.")

    output_dir.mkdir(parents=True, exist_ok=True)

    interval = video_fps / frames_per_second
    frame_index = 0
    saved_count = 0

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_index / video_fps
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        milliseconds = int((time_sec - int(time_sec)) * 1000)

        frame_name = f"{minutes:02d}_{seconds:02d}_{milliseconds:03d}.png"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), frame)
        saved_count += 1

        frame_index += interval

    cap.release()
    print(f"  - Done: {saved_count} frames saved to {output_dir}")

def main():
    # Load configuration from config_stage1.json
    with open("config_stage1.json", "r") as f:
        config = json.load(f)

    fps = 1.0  # Default frames per second

    video_root_dir = Path("data/video")
    output_root_dir = Path(config["paths"]["unlabeled_dir"])

    if not video_root_dir.is_dir():
        print(f"Error: Video root directory not found: {video_root_dir}")
        return

    model_types = [d.name for d in video_root_dir.iterdir() if d.is_dir()]
    if not model_types:
        print(f"No model types found in {video_root_dir}")
        return
        
    print(f"Found model types to extract: {model_types}")

    for data_type in model_types:
        video_dir = video_root_dir / data_type
        output_dir = output_root_dir / data_type

        print(f"\n========================================")
        print(f"🚀 STARTING EXTRACTION FOR: {data_type.upper()}")
        print(f"========================================")

        if not video_dir.is_dir():
            print(f"Error: Video directory not found: {video_dir}")
            continue

        video_files_found = False
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith((".mp4", ".avi")):
                    video_files_found = True
                    video_path = Path(root) / file
                    relative_path = video_path.relative_to(video_dir)
                    frame_output_dir = output_dir / relative_path.parent / video_path.stem
                    extract_frames(video_path, frame_output_dir, args.fps)
        
        if not video_files_found:
            print(f"No video files (.mp4, .avi) found in {video_dir}")

    print("\n✅ All extractions finished.")

if __name__ == "__main__":
    main()
