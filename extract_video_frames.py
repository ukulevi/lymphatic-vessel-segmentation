#!/usr/bin/env python3
"""
Extract frames from video file.
"""
import cv2
import os
import argparse

def extract_frames(video_path: str, output_dir: str, num_frames: int = 1000):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract
    """
    print(f"📹 Extracting {num_frames} frames from {video_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Total frames in video: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Calculate step size to extract evenly distributed frames
    if num_frames >= total_frames:
        step = 1
        num_frames = total_frames
        print(f"  ⚠️  Requested {num_frames} frames, but video only has {total_frames}. Extracting all frames.")
    else:
        step = max(1, total_frames // num_frames)
        print(f"  Extracting every {step} frames to get {num_frames} frames")
    
    saved = 0
    frame_count = 0
    
    while saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it matches step
        if frame_count % step == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved += 1
            
            if saved % 100 == 0:
                print(f"  ✓ Saved {saved}/{num_frames} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {saved} frames to {output_dir}")
    return saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=str, default="data/video/Human_Lymphatic_02-12-24_pressure_0ca_scan_East2.mp4",
                       help="Path to video file")
    parser.add_argument("--output", type=str, default="data/video_frames",
                       help="Output directory for frames")
    parser.add_argument("--num_frames", type=int, default=1000,
                       help="Number of frames to extract")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.num_frames)

