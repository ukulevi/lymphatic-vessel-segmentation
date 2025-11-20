#!/usr/bin/env python
"""
Script to convert LabelMe JSON annotations to binary masks.
Usage:
    python convert_json_to_mask.py
"""
import os
import cv2
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def convert_json_to_mask(json_path, image_shape):
    """
    Convert LabelMe JSON annotation to binary mask.
    Args:
        json_path: Path to JSON file
        image_shape: Tuple of (height, width)
    Returns:
        Binary mask as uint8 array
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {json_path}: {e}")
        return np.zeros(image_shape, dtype=np.uint8)

    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for shape in data.get('shapes', []):
        if shape.get('shape_type') in (None, 'polygon'):
            points = np.array(shape.get('points', []), dtype=np.int32)
            if points.size > 0 and points.ndim == 2 and points.shape[1] == 2:
                cv2.fillPoly(mask, [points], color=255)

    return mask

def process_directory(input_dir, output_dir):
    """
    Process all JSON files in input_dir and save masks to output_dir.
    Maintains folder structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all JSON files
    json_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.json'):
                rel_path = os.path.relpath(root, input_dir)
                json_files.append((
                    os.path.join(root, f),
                    os.path.join(output_dir, rel_path)
                ))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    for json_path, out_dir in tqdm(json_files):
        # Get corresponding image path
        img_path = os.path.splitext(json_path)[0] + '.png'
        if not os.path.exists(img_path):
            img_path = os.path.splitext(json_path)[0] + '.jpg'
        
        if not os.path.exists(img_path):
            print(f"No image found for {json_path}")
            continue
            
        # Read image to get shape
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_path}")
            continue
            
        # Convert annotation to mask
        mask = convert_json_to_mask(json_path, image.shape[:2])
        
        # Save mask
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(json_path))[0] + '_mask.png'
        )
        cv2.imwrite(out_path, mask)

def main():
    # Load config to get data type
    with open("config.json", "r") as f:
        config = json.load(f)
    data_type = config.get("type", "Human")

    input_dir = os.path.join("data", "annotated", data_type)
    output_dir = os.path.join("data", "masks", data_type)
    
    process_directory(input_dir, output_dir)

if __name__ == '__main__':
    main()