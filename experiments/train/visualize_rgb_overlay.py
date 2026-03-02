#!/usr/bin/env python3
"""
Visualize PhysTwin predictions overlaid on masked RGB cloth images.

Shows predicted 3D points projected onto actual cloth photos at time T+1.
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_camera_calibration(calibration_dir):
    """Load camera intrinsics and extrinsics from calibration directory."""
    calib_dir = Path(calibration_dir)

    # Load intrinsics
    with open(calib_dir / 'base.pkl', 'rb') as f:
        base = pickle.load(f)

    # Load extrinsics
    with open(calib_dir / 'rvecs.pkl', 'rb') as f:
        rvecs = pickle.load(f)
    with open(calib_dir / 'tvecs.pkl', 'rb') as f:
        tvecs = pickle.load(f)

    # Get camera serial numbers
    cam_serials = sorted(rvecs.keys())

    # Extract parameters for each camera
    from scipy.spatial.transform import Rotation

    cameras = {}
    for serial in cam_serials:
        # Intrinsics
        if serial in base and 'mtx' in base[serial]:
            K = base[serial]['mtx']
        else:
            K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)

        # Extrinsics (world to camera)
        rvec = rvecs[serial]
        tvec = tvecs[serial]
        R = Rotation.from_rotvec(rvec.flatten()).as_matrix()

        # Build 4x4 transform
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = tvec.flatten()

        cameras[serial] = {'K': K, 'E': E, 'R': R, 't': tvec}

    return cameras, cam_serials


def project_points_to_image(points_3d, K, R, t):
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) 3D points in world coordinates
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix (world to camera)
        t: (3, 1) translation vector (world to camera)

    Returns:
        points_2d: (N, 2) 2D image coordinates
        valid_mask: (N,) boolean mask of points in front of camera
    """
    # Transform to camera coordinates
    points_cam = (R @ points_3d.T).T + t.T

    # Only keep points in front of camera
    valid_mask = points_cam[:, 2] > 0

    # Project to image plane
    points_2d_homog = (K @ points_cam.T).T
    points_2d = points_2d_homog[:, :2] / points_2d_homog[:, 2:3]

    return points_2d, valid_mask


def load_image_and_mask(rgb_path, mask_path):
    """Load RGB image and mask."""
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = mask > 127  # Binary mask

    return rgb, mask


def create_overlay_visualization(rgb, mask, points_2d, valid_mask, point_color, title):
    """Create visualization with points overlaid on masked RGB image."""
    # Apply mask to RGB
    masked_rgb = rgb.copy()
    masked_rgb[~mask] = masked_rgb[~mask] // 3  # Dim non-cloth regions

    # Draw points
    viz = masked_rgb.copy()
    h, w = rgb.shape[:2]

    for pt, valid in zip(points_2d, valid_mask):
        if valid:
            x, y = int(pt[0]), int(pt[1])
            # Check if point is within image bounds
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(viz, (x, y), 2, point_color, -1)

    return viz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to PGND processed episode directory')
    parser.add_argument('--predictions_pkl', type=str, required=True,
                        help='Path to predictions pickle file')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Which camera to use (0-3)')
    parser.add_argument('--test_frame_idx', type=int, default=15,
                        help='Which test frame to visualize')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')

    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {episode_dir}")

    # Load camera calibration
    print("Loading camera calibration...")
    cameras, cam_serials = load_camera_calibration(episode_dir / 'calibration')

    # Get camera for visualization
    cam_serial = cam_serials[args.camera_idx]
    cam = cameras[cam_serial]
    K, R, t = cam['K'], cam['R'], cam['t']

    print(f"Using camera {args.camera_idx} (serial: {cam_serial})")

    # Load predictions
    print(f"Loading predictions from: {args.predictions_pkl}")
    with open(args.predictions_pkl, 'rb') as f:
        results = pickle.load(f)

    predictions = results['predictions']  # List of (N, 3) arrays
    ground_truth = results['ground_truth']  # List of (N, 3) arrays

    # Get the requested test frame
    if args.test_frame_idx >= len(predictions):
        print(f"Warning: test_frame_idx {args.test_frame_idx} >= num predictions {len(predictions)}")
        args.test_frame_idx = len(predictions) // 2
        print(f"Using middle frame instead: {args.test_frame_idx}")

    pred_points = predictions[args.test_frame_idx]  # (N, 3)
    gt_points = ground_truth[args.test_frame_idx]  # (N, 3)

    # Load RGB and mask for frame T+1
    # Note: predictions[i] is for time T+1 where i is the test frame index
    # We need to map this to the actual frame number in the episode

    # Load split to get actual frame numbers
    split_file = episode_dir.parent / 'split.json'
    if split_file.exists():
        import json
        with open(split_file) as f:
            split = json.load(f)
        test_start = split['test'][0]
    else:
        # Assume 70/30 split
        rgb_dir = episode_dir / f'camera_{args.camera_idx}' / 'rgb'
        total_frames = len(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
        test_start = int(total_frames * 0.7)

    actual_frame = test_start + args.test_frame_idx + 1  # +1 for T+1

    # Load RGB and mask
    rgb_dir = episode_dir / f'camera_{args.camera_idx}' / 'rgb'
    mask_dir = episode_dir / f'camera_{args.camera_idx}' / 'mask'

    rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
    mask_files = sorted(mask_dir.glob('*.png'))

    if actual_frame >= len(rgb_files):
        print(f"Warning: frame {actual_frame} out of range, using frame {len(rgb_files)-1}")
        actual_frame = len(rgb_files) - 1

    print(f"Loading RGB and mask for frame {actual_frame}")
    rgb, mask = load_image_and_mask(rgb_files[actual_frame], mask_files[actual_frame])

    print(f"RGB shape: {rgb.shape}, Mask shape: {mask.shape}")
    print(f"Predicted points shape: {pred_points.shape}")
    print(f"Ground truth points shape: {gt_points.shape}")

    # Project points to image
    print("Projecting 3D points to 2D...")
    pred_2d, pred_valid = project_points_to_image(pred_points, K, R, t)
    gt_2d, gt_valid = project_points_to_image(gt_points, K, R, t)

    print(f"Valid predicted points: {pred_valid.sum()}/{len(pred_valid)}")
    print(f"Valid ground truth points: {gt_valid.sum()}/{len(gt_valid)}")

    # Create visualizations
    print("Creating overlay visualizations...")

    # GT overlay (green)
    gt_viz = create_overlay_visualization(rgb, mask, gt_2d, gt_valid,
                                          (0, 255, 0), "Ground Truth T+1")

    # Predicted overlay (red)
    pred_viz = create_overlay_visualization(rgb, mask, pred_2d, pred_valid,
                                            (255, 0, 0), "Predicted T+1")

    # Both overlaid (green=GT, red=Pred)
    both_viz = rgb.copy()
    both_viz[~mask] = both_viz[~mask] // 3
    h, w = rgb.shape[:2]

    # Draw GT in green
    for pt, valid in zip(gt_2d, gt_valid):
        if valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(both_viz, (x, y), 2, (0, 255, 0), -1)

    # Draw predictions in red
    for pt, valid in zip(pred_2d, pred_valid):
        if valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(both_viz, (x, y), 2, (255, 0, 0), -1)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Masked RGB
    masked_display = rgb.copy()
    masked_display[~mask] = masked_display[~mask] // 3
    axes[0, 0].imshow(masked_display)
    axes[0, 0].set_title(f'Masked RGB (Frame {actual_frame})', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # GT overlay
    axes[0, 1].imshow(gt_viz)
    axes[0, 1].set_title('Ground Truth Points at T+1 (Green)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Predicted overlay
    axes[1, 0].imshow(pred_viz)
    axes[1, 0].set_title('Predicted Points at T+1 (Red)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Both overlaid
    axes[1, 1].imshow(both_viz)
    axes[1, 1].set_title('Overlay: GT (Green) vs Predicted (Red)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(f'RGB Overlay Visualization - Test Frame {args.test_frame_idx} (Actual Frame {actual_frame})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_file = output_dir / f'rgb_overlay_frame_{args.test_frame_idx:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")

    plt.close()


if __name__ == '__main__':
    main()
