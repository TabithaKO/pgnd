#!/usr/bin/env python3
"""
Convert PGND data format to PhysTwin data format for testing
PhysTwin optimization and future prediction on custom cloth data.

Usage:
    python convert_pgnd_to_phystwin.py \
        --pgnd_episode /home/fashionista/pgnd/experiments/log/data_cloth/1224_cloth_fold_processed/episode_0000 \
        --output_dir /home/fashionista/PhysTwin/data/pgnd_cloth \
        --case_name pgnd_cloth_episode0
"""

import os
import sys
import json
import shutil
import pickle
import argparse
import numpy as np
from pathlib import Path


def convert_calibration(pgnd_calib_dir, output_file):
    """
    Convert PGND calibration format to PhysTwin calibrate.pkl format.

    PhysTwin expects:
    {
        'cam_intrinsics': [K0, K1, K2, ...],  # 3x3 camera matrices
        'cam_extrinsics': [E0, E1, E2, ...],  # 4x4 world-to-camera transforms
    }
    """
    # Load PGND calibration
    base_path = os.path.join(pgnd_calib_dir, 'base.pkl')
    rvecs_path = os.path.join(pgnd_calib_dir, 'rvecs.pkl')
    tvecs_path = os.path.join(pgnd_calib_dir, 'tvecs.pkl')

    with open(base_path, 'rb') as f:
        base = pickle.load(f)
    with open(rvecs_path, 'rb') as f:
        rvecs = pickle.load(f)
    with open(tvecs_path, 'rb') as f:
        tvecs = pickle.load(f)

    # Extract camera intrinsics and extrinsics
    cam_intrinsics = []
    cam_extrinsics = []

    # Get camera serial numbers (rvecs/tvecs are keyed by serial numbers)
    cam_serials = sorted(rvecs.keys())
    print(f"  Camera serials: {cam_serials}")

    # Import scipy for rotation conversion
    from scipy.spatial.transform import Rotation

    # PGND calibration is keyed by camera serial numbers
    for cam_idx, serial in enumerate(cam_serials):
        # Intrinsics (3x3 matrix)
        if serial in base:
            cam_data = base[serial]
            if 'mtx' in cam_data:  # PGND stores intrinsics as 'mtx'
                K = cam_data['mtx']
            elif 'intrinsics' in cam_data:
                K = cam_data['intrinsics']
            else:
                # Default intrinsics
                K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)
                print(f"  Warning: Using default intrinsics for camera {cam_idx} ({serial})")
        else:
            # Default intrinsics if not found
            K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)
            print(f"  Warning: Using default intrinsics for camera {cam_idx} ({serial})")

        cam_intrinsics.append(K)

        # Extrinsics (4x4 world-to-camera transform)
        rvec = rvecs[serial]
        tvec = tvecs[serial]

        # Convert rotation vector to rotation matrix
        R = Rotation.from_rotvec(rvec.flatten()).as_matrix()

        # Build 4x4 transform
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = tvec.flatten()
        cam_extrinsics.append(E)

    # Save in PhysTwin format
    calibration = {
        'cam_intrinsics': cam_intrinsics,
        'cam_extrinsics': cam_extrinsics,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(calibration, f)

    print(f"Converted calibration: {len(cam_intrinsics)} cameras")
    return len(cam_intrinsics)


def create_metadata(num_cameras, output_file):
    """
    Create metadata.json for PhysTwin.
    """
    metadata = {
        'num_cameras': num_cameras,
        'fps': 30,  # Adjust if your data has different FPS
        'image_width': 640,  # Adjust based on your image size
        'image_height': 480,
        'data_type': 'real',  # or 'sim'
    }

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata: {num_cameras} cameras")


def convert_episode(pgnd_episode_dir, output_dir, case_name):
    """
    Convert one PGND episode to PhysTwin format.
    """
    pgnd_path = Path(pgnd_episode_dir)
    output_path = Path(output_dir) / case_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting PGND episode:")
    print(f"  Source: {pgnd_path}")
    print(f"  Output: {output_path}")

    # Find all camera directories
    camera_dirs = sorted([d for d in pgnd_path.iterdir() if d.name.startswith('camera_')])
    num_cameras = len(camera_dirs)
    print(f"  Found {num_cameras} cameras")

    # Create output directories for each camera view
    for cam_idx, cam_dir in enumerate(camera_dirs):
        print(f"\n  Processing {cam_dir.name}...")

        # Copy RGB images (rename to color)
        src_rgb = cam_dir / 'rgb'
        dst_color = output_path / 'color' / str(cam_idx)
        dst_color.mkdir(parents=True, exist_ok=True)

        if src_rgb.exists():
            rgb_files = sorted(src_rgb.glob('*.png')) + sorted(src_rgb.glob('*.jpg'))
            for img_file in rgb_files:
                shutil.copy(img_file, dst_color / img_file.name)
            print(f"    Copied {len(rgb_files)} RGB images")

        # Copy depth images
        src_depth = cam_dir / 'depth'
        dst_depth = output_path / 'depth' / str(cam_idx)
        dst_depth.mkdir(parents=True, exist_ok=True)

        if src_depth.exists():
            depth_files = sorted(src_depth.glob('*.png')) + sorted(src_depth.glob('*.npy'))
            for img_file in depth_files:
                shutil.copy(img_file, dst_depth / img_file.name)
            print(f"    Copied {len(depth_files)} depth images")

        # Copy mask images
        src_mask = cam_dir / 'mask'
        dst_mask = output_path / 'mask' / str(cam_idx)
        dst_mask.mkdir(parents=True, exist_ok=True)

        if src_mask.exists():
            mask_files = sorted(src_mask.glob('*.png'))
            for img_file in mask_files:
                shutil.copy(img_file, dst_mask / img_file.name)
            print(f"    Copied {len(mask_files)} mask images")

    # Convert calibration
    calib_dir = pgnd_path / 'calibration'
    calib_output = output_path / 'calibrate.pkl'
    num_cams = convert_calibration(str(calib_dir), str(calib_output))

    # Create metadata
    metadata_output = output_path / 'metadata.json'
    create_metadata(num_cams, str(metadata_output))

    print(f"\n✅ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Process data:")
    print(f"   cd /home/fashionista/PhysTwin")
    print(f"   python data_process/process_single_case.py --case_name {case_name}")
    print(f"\n2. Optimize physics (Stage 1):")
    print(f"   python optimize_warp.py --base_path {output_dir} --case_name {case_name}")
    print(f"\n3. Train dynamics (Stage 2):")
    print(f"   python train_warp.py --base_path {output_dir} --case_name {case_name} --train_frame 80")
    print(f"\n4. Test future prediction:")
    print(f"   python inference_warp.py --base_path {output_dir} --case_name {case_name}")


def main():
    parser = argparse.ArgumentParser(description='Convert PGND data to PhysTwin format')
    parser.add_argument('--pgnd_episode', type=str, required=True,
                        help='Path to PGND episode directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PhysTwin data')
    parser.add_argument('--case_name', type=str, required=True,
                        help='Name for this PhysTwin case')

    args = parser.parse_args()

    convert_episode(args.pgnd_episode, args.output_dir, args.case_name)


if __name__ == '__main__':
    main()
