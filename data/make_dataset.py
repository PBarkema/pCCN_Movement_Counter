

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm


def download_videos_from_s3():
    """ 
        Downloads all videos from the specified S3 bucket and prefix to a local directory.
    """
    # Connect to S3 without authentication (public bucket)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    bucket_name = 'prism-mvta'
    prefix = 'training-and-validation-data/'
    download_dir = './raw'

    os.makedirs(download_dir, exist_ok=True)

    # List all objects in the S3 path
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    video_names = []

    for page in pages:
        if 'Contents' not in page:
            print("No files found at the specified path!")
            break

        for obj in page['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)

            if not filename:
                continue

            video_names.append(filename)

            local_path = os.path.join(download_dir, filename)
            print(f"Downloading: {filename}")
            s3.download_file(bucket_name, key, local_path)

    print("\n" + "="*50)
    print("Downloaded videos:")
    print("="*50)
    for name in video_names:
        print(name)

    print(f"\nTotal: {len(video_names)} files")





def extract_keypoints_with_distances(video_path, model, target_length=300):
    """
    Extracts keypoints AND computes pairwise geometric distances.

    Returns tensor of shape (target_length, 323).
    Structure: [34 raw coords] + [289 pairwise distances]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return torch.zeros(target_length, 323)

    # Run YOLO inference
    results = model(frames, verbose=False, stream=True)

    frame_features = []

    for r in results:
        # 1. Handle Empty Detection
        if r.keypoints.xyn.shape[0] == 0:
            # If no person detected, fill with zeros
            frame_features.append(torch.zeros(34 + 17*17))
            continue

        # 2. Get Normalized Coordinates (17 keypoints, 2 coords)
        # Shape: (17, 2)
        kpts_xy = r.keypoints.xyn[0].cpu()

        # 3. Compute Distance Matrix
        # shape: (17, 17) -> Distance between every pair of joints
        # p=2 is Euclidean distance
        dist_matrix = torch.cdist(kpts_xy, kpts_xy, p=2)

        # 4. Flatten everything
        flat_coords = kpts_xy.view(-1)      # (34,)
        flat_dists = dist_matrix.view(-1)   # (289,)

        # 5. Concatenate
        # Total size: 34 + 289 = 323
        combined = torch.cat([flat_coords, flat_dists])

        frame_features.append(combined)

    if len(frame_features) == 0:
        return torch.zeros(target_length, 323)

    # Stack into (T_original, 323)
    seq = torch.stack(frame_features)

    # Resample to fixed length (300)
    original_len = seq.shape[0]
    indices = torch.linspace(0, original_len - 1, target_length).long()
    seq = seq[indices]

    return seq

def feature_engineering(video_dir, save_path, MODEL_NAME='yolov8n-pose.pt'):
    model = YOLO(MODEL_NAME)

    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} not found.")
        return

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    data = []
    labels = []
    filenames = []

    print(f"Processing {len(video_files)} videos...")
    print("Features: 34 Coordinates + 289 Distances = 323 Features per frame")

    for f in tqdm(video_files):
        path = os.path.join(video_dir, f)

        # Parse label (Assumes format: "3_pushups.mp4")
        try:
            label = int(f.split('_')[0])
        except ValueError:
            print(f"Skipping {f}: Could not extract label.")
            continue

        # Extract Enhanced Features
        features = extract_keypoints_with_distances(path, model, target_length=300)

        data.append(features)
        labels.append(label)
        filenames.append(f)

    # Save everything
    torch.save({
        "data": torch.stack(data),         # Shape: (N, 300, 323)
        "labels": torch.tensor(labels),    # Shape: (N,)
        "filenames": filenames             # List of strings
    }, save_path)

    print(f"Success! Saved to {save_path}")
    print(f"Final Data Shape: {torch.stack(data).shape}")

# --- RUN IT ---
