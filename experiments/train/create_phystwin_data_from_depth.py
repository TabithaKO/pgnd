#!/usr/bin/env python3
"""
Create PhysTwin final_data.pkl from PGND RGB-D data without running full CoTracker pipeline.
Uses depth maps to generate 3D point clouds directly.
"""

import os
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def load_depth_image(depth_path):
    """Load depth image (PNG or NPY format)."""
    if depth_path.suffix == '.npy':
        return np.load(depth_path)
    else:
        # PNG depth (usually in mm, convert to meters)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 1000.0  # mm to meters


def depth_to_pointcloud(depth, mask, K, max_points=2000):
    """
    Convert depth map to 3D point cloud.

    Args:
        depth: (H, W) depth map in meters
        mask: (H, W) binary mask of cloth region
        K: (3, 3) camera intrinsics matrix
        max_points: Maximum number of points to sample

    Returns:
        points: (N, 3) 3D points in camera coordinates
        valid: (N,) validity flags
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create meshgrid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Apply mask
    valid_mask = (mask > 0) & (depth > 0) & (depth < 5.0)  # Valid depth range
    u = u[valid_mask]
    v = v[valid_mask]
    z = depth[valid_mask]

    # Unproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=1)  # (N, 3)

    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    valid = np.ones(len(points), dtype=bool)

    return points, valid


def create_final_data_pkl(phystwin_case_dir, num_frames=100, max_points=2000):
    """
    Create PhysTwin's final_data.pkl from converted PGND data.

    Args:
        phystwin_case_dir: Path to PhysTwin case directory
        num_frames: Number of frames to process (default 100 for quick test)
        max_points: Maximum points per frame
    """
    case_path = Path(phystwin_case_dir)

    print(f"Creating final_data.pkl for {case_path.name}")
    print(f"  Processing {num_frames} frames")

    # Load calibration
    calib_path = case_path / 'calibrate.pkl'
    with open(calib_path, 'rb') as f:
        calib = pickle.load(f)

    cam_intrinsics = calib['cam_intrinsics']
    cam_extrinsics = calib['cam_extrinsics']
    num_cameras = len(cam_intrinsics)

    print(f"  Found {num_cameras} cameras")

    # Use camera 0 for simplicity (PhysTwin can use multi-view later)
    camera_idx = 0
    K = cam_intrinsics[camera_idx]
    E = cam_extrinsics[camera_idx]

    # Get image paths
    color_dir = case_path / 'color' / str(camera_idx)
    depth_dir = case_path / 'depth' / str(camera_idx)
    mask_dir = case_path / 'mask' / str(camera_idx)

    # Try both PNG and JPG for RGB
    color_files = sorted(list(color_dir.glob('*.png')) + list(color_dir.glob('*.jpg')))[:num_frames]
    depth_files = sorted(depth_dir.glob('*.*'))[:num_frames]
    mask_files = sorted(mask_dir.glob('*.png'))[:num_frames]

    if len(color_files) != len(depth_files) or len(color_files) != len(mask_files):
        print(f"Warning: Mismatched file counts: RGB={len(color_files)}, Depth={len(depth_files)}, Mask={len(mask_files)}")

    num_frames = min(len(color_files), len(depth_files), len(mask_files))
    print(f"  Processing {num_frames} frames")

    # Process each frame
    all_points = []
    all_valid = []

    for i in tqdm(range(num_frames), desc="Processing frames"):
        # Load depth and mask
        depth = load_depth_image(depth_files[i])
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)

        # Generate point cloud
        points, valid = depth_to_pointcloud(depth, mask, K, max_points=max_points)

        # Transform to world coordinates using extrinsics
        # E is world-to-camera, so we need camera-to-world (inverse)
        E_inv = np.linalg.inv(E)
        points_homog = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
        points_world = (E_inv @ points_homog.T).T[:, :3]  # (N, 3)

        all_points.append(points_world)
        all_valid.append(valid)

    # Pad to consistent number of points
    max_pts = max([len(pts) for pts in all_points])
    padded_points = np.zeros((num_frames, max_pts, 3))
    padded_valid = np.zeros((num_frames, max_pts), dtype=bool)

    for i in range(num_frames):
        n = len(all_points[i])
        padded_points[i, :n] = all_points[i]
        padded_valid[i, :n] = all_valid[i]

    print(f"  Point cloud shape: {padded_points.shape}")

    # Create final_data.pkl in PhysTwin format
    final_data = {
        'gt_object_points': padded_points,  # (T, N, 3)
        'gt_object_visibilities': padded_valid,  # (T, N)
        'gt_object_motions_valid': padded_valid,  # (T, N)
        'cam_intrinsics': cam_intrinsics,
        'cam_extrinsics': cam_extrinsics,
        'num_frames': num_frames,
        'num_points': max_pts,
    }

    # Save
    output_path = case_path / 'final_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"✅ Created: {output_path}")

    # Also create split.json (train/test split)
    train_end = int(num_frames * 0.7)
    split = {
        'train': [0, train_end],
        'test': [train_end, num_frames]
    }

    split_path = case_path / 'split.json'
    import json
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"✅ Created: {split_path}")
    print(f"   Train: frames 0-{train_end}")
    print(f"   Test: frames {train_end}-{num_frames}")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_dir', type=str, required=True,
                        help='PhysTwin case directory')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Number of frames to process (default 100)')
    parser.add_argument('--max_points', type=int, default=2000,
                        help='Max points per frame (default 2000)')

    args = parser.parse_args()

    create_final_data_pkl(args.case_dir, args.num_frames, args.max_points)


if __name__ == '__main__':
    main()
