#!/bin/bash
# Run PhysTwin on a single processed PGND episode with RGB overlay visualization

set -e

echo "========================================================================"
echo "PhysTwin on Processed PGND Episode - Full Pipeline with RGB Overlay"
echo "========================================================================"
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PGND_EPISODE="/home/fashionista/pgnd/experiments/log/data_cloth/0114_cloth6_processed/episode_0000"
PHYSTWIN_BASE="/home/fashionista/PhysTwin"
CASE_NAME="pgnd_processed_rgb_${TIMESTAMP}"
PHYSTWIN_DATA="$PHYSTWIN_BASE/data/$CASE_NAME"
RESULTS_DIR="$PHYSTWIN_BASE/${CASE_NAME}"

# Pretrained parameters
PRETRAINED_PARAMS="/home/fashionista/PhysTwin/experiments_optimization/single_lift_cloth_3/optimal_params.pkl"

echo "📁 Episode: $PGND_EPISODE"
echo "📊 Output: $RESULTS_DIR"
echo ""

# Create results directories
mkdir -p "$RESULTS_DIR/rgb_overlays"
mkdir -p "$RESULTS_DIR/logs"

# Check episode structure
if [ ! -d "$PGND_EPISODE/camera_0" ]; then
    echo "❌ Error: Episode missing camera data"
    exit 1
fi

echo "✓ Found camera_0, camera_1, camera_2, camera_3"
echo "✓ Found calibration/"
echo ""

# Step 1: Convert processed episode to PhysTwin format
echo "========================================================================"
echo "Step 1: Convert RGB-D data to PhysTwin format"
echo "========================================================================"
echo ""

# Export paths for Python script
export PGND_EPISODE_PATH="$PGND_EPISODE"
export PHYSTWIN_OUTPUT_PATH="$PHYSTWIN_DATA"

python << 'CONVERT_SCRIPT'
import pickle
import numpy as np
import json
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

episode_path = Path(os.environ['PGND_EPISODE_PATH'])
output_path = Path(os.environ['PHYSTWIN_OUTPUT_PATH'])
output_path.mkdir(parents=True, exist_ok=True)

print("Loading camera calibration...")

# Load camera calibration (same as visualize_rgb_overlay.py)
calib_dir = episode_path / 'calibration'
with open(calib_dir / 'base.pkl', 'rb') as f:
    base = pickle.load(f)
with open(calib_dir / 'rvecs.pkl', 'rb') as f:
    rvecs = pickle.load(f)
with open(calib_dir / 'tvecs.pkl', 'rb') as f:
    tvecs = pickle.load(f)

cam_serials = sorted(rvecs.keys())
print(f"Found {len(cam_serials)} cameras: {cam_serials}")

# Use camera 0 for point cloud generation
camera_idx = 0
cam_serial = cam_serials[camera_idx]

# Get intrinsics
K = base[cam_serial]['mtx'] if cam_serial in base and 'mtx' in base[cam_serial] else \
    np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)

# Get extrinsics (world to camera)
rvec = rvecs[cam_serial]
tvec = tvecs[cam_serial]
R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
E = np.eye(4)
E[:3, :3] = R
E[:3, 3] = tvec.flatten()

print(f"Using camera {camera_idx} (serial: {cam_serial})")
print(f"Intrinsics K:\n{K}")
print(f"Extrinsics E (world->cam):\n{E}")

# Load depth and mask files
rgb_dir = episode_path / f'camera_{camera_idx}' / 'rgb'
depth_dir = episode_path / f'camera_{camera_idx}' / 'depth'
mask_dir = episode_path / f'camera_{camera_idx}' / 'mask'

rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
depth_files = sorted(depth_dir.glob('*.png'))
mask_files = sorted(mask_dir.glob('*.png'))

num_frames = min(len(rgb_files), len(depth_files), len(mask_files))
print(f"\nProcessing {num_frames} frames...")

# Convert depth to point clouds
def depth_to_pointcloud(depth, mask, K, max_points=8000):
    """Convert depth map to 3D point cloud in camera coordinates."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create meshgrid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Apply mask
    valid_mask = (mask > 127) & (depth > 0) & (depth < 5000)  # depth in mm
    u = u[valid_mask]
    v = v[valid_mask]
    z = depth[valid_mask].astype(np.float32) / 1000.0  # mm to meters

    # Unproject to 3D (camera coordinates)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1)  # (N, 3) in camera coords

    # Subsample if too many points
    if len(points_cam) > max_points:
        indices = np.random.choice(len(points_cam), max_points, replace=False)
        points_cam = points_cam[indices]

    return points_cam

all_points_world = []

for i in tqdm(range(num_frames), desc="Converting frames"):
    # Load depth and mask
    depth = cv2.imread(str(depth_files[i]), cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)

    # Generate point cloud in camera coordinates
    points_cam = depth_to_pointcloud(depth, mask, K, max_points=8000)

    # Transform to world coordinates (inverse of extrinsics)
    E_inv = np.linalg.inv(E)
    points_homog = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (E_inv @ points_homog.T).T[:, :3]

    all_points_world.append(points_world)

# Pad to consistent number of points
max_pts = max([len(pts) for pts in all_points_world])
padded_points = np.zeros((num_frames, max_pts, 3))
padded_valid = np.zeros((num_frames, max_pts), dtype=bool)

for i in range(num_frames):
    n = len(all_points_world[i])
    padded_points[i, :n] = all_points_world[i]
    padded_valid[i, :n] = True

print(f"\nPoint cloud shape: {padded_points.shape}")
print(f"  {num_frames} frames")
print(f"  {max_pts} points per frame")

# Create PhysTwin final_data.pkl
final_data = {
    'object_points': padded_points,
    'object_visibilities': padded_valid,
    'object_motions_valid': padded_valid,
    'object_colors': np.ones((num_frames, max_pts, 3)) * 0.5,
}

output_file = output_path / 'final_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(final_data, f)

print(f"✓ Created: {output_file}")

# Create train/test split (70/30)
train_end = int(num_frames * 0.7)
split = {
    'train': list(range(0, train_end)),
    'test': list(range(train_end, num_frames))
}

split_file = output_path / 'split.json'
with open(split_file, 'w') as f:
    json.dump(split, f, indent=2)

print(f"✓ Created: {split_file}")
print(f"  Train: frames 0-{train_end}")
print(f"  Test: frames {train_end}-{num_frames}")

# Save metadata including calibration for later visualization
metadata = {
    'num_frames': num_frames,
    'num_points': max_pts,
    'source_episode': str(episode_path),
    'camera_idx': camera_idx,
    'camera_serial': cam_serial,
}

metadata_file = output_path / 'metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Created: {metadata_file}")

CONVERT_SCRIPT

echo ""
echo "✅ Data conversion complete!"
echo ""

# Step 2: Copy pretrained parameters for warm start
echo "========================================================================"
echo "Step 2: Initialize with pretrained cloth parameters"
echo "========================================================================"
echo ""

PARAMS_DIR="/home/fashionista/PhysTwin/experiments_optimization/$CASE_NAME"
mkdir -p "$PARAMS_DIR"

if [ -f "$PRETRAINED_PARAMS" ]; then
    cp "$PRETRAINED_PARAMS" "$PARAMS_DIR/optimal_params.pkl"
    echo "✓ Copied pretrained parameters to $PARAMS_DIR/optimal_params.pkl"

    export PARAMS_FILE="$PARAMS_DIR/optimal_params.pkl"
    python << 'SHOW_PARAMS'
import pickle
import os
params_file = os.environ['PARAMS_FILE']
with open(params_file, 'rb') as f:
    params = pickle.load(f)
print("\nInitial parameters:")
for key, val in params.items():
    print(f"  {key}: {val}")
SHOW_PARAMS
else
    echo "⚠️  Pretrained parameters not found, will start from scratch"
fi

echo ""

# Step 3: Train PhysTwin
echo "========================================================================"
echo "Step 3: Train PhysTwin (Physics Optimization)"
echo "========================================================================"
echo ""

cd /home/fashionista/PhysTwin

# Get train_frame count
TRAIN_FRAME=$(python3 -c "import json; split = json.load(open('$PHYSTWIN_DATA/split.json')); print(len(split['train']))")

echo "Training on $TRAIN_FRAME frames..."
echo ""

python train_warp.py \
  --base_path "$PHYSTWIN_BASE" \
  --case_name "$CASE_NAME" \
  --train_frame "$TRAIN_FRAME"

echo ""
echo "✅ Training complete!"
echo ""

# Step 4: Run inference
echo "========================================================================"
echo "Step 4: Run Inference on Test Frames"
echo "========================================================================"
echo ""

python inference_warp.py \
  --base_path "$PHYSTWIN_BASE" \
  --case_name "$CASE_NAME"

echo ""
echo "✅ Inference complete!"
echo ""

# Step 5: Create RGB overlay visualizations
echo "========================================================================"
echo "Step 5: Create RGB Overlay Visualizations"
echo "========================================================================"
echo ""

# Find the predictions file
PREDICTIONS_PKL="/home/fashionista/PhysTwin/experiments_optimization/$CASE_NAME/predictions.pkl"
if [ ! -f "$PREDICTIONS_PKL" ]; then
    echo "⚠️  Predictions file not found: $PREDICTIONS_PKL"
    echo "   Checking alternative locations..."
    PREDICTIONS_PKL=$(find /home/fashionista/PhysTwin/experiments* -name "*${CASE_NAME}*predictions.pkl" -o -name "*${CASE_NAME}*inference.pkl" 2>/dev/null | head -1)
fi

if [ -f "$PREDICTIONS_PKL" ]; then
    echo "Found predictions: $PREDICTIONS_PKL"
    echo ""

    OUTPUT_VIZ_DIR="$RESULTS_DIR/rgb_overlays"

    # Visualize multiple test frames (every 3rd)
    for frame_idx in 0 3 6 9 12; do
        echo "Visualizing test frame $frame_idx..."
        python /home/fashionista/pgnd/experiments/train/visualize_rgb_overlay.py \
            --episode_dir "$PGND_EPISODE" \
            --predictions_pkl "$PREDICTIONS_PKL" \
            --camera_idx 0 \
            --test_frame_idx "$frame_idx" \
            --output_dir "$OUTPUT_VIZ_DIR" || true
    done

    echo ""
    echo "✅ RGB overlay visualizations complete!"
    echo "   Output: $OUTPUT_VIZ_DIR"
else
    echo "❌ Could not find predictions file"
fi

echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "📁 Outputs:"
echo "   Data: $PHYSTWIN_DATA"
echo "   Model: /home/fashionista/PhysTwin/experiments_optimization/$CASE_NAME"
echo "   Visualizations: /home/fashionista/pgnd/experiments/train/viz_rgb_overlay_${CASE_NAME}"
echo ""
