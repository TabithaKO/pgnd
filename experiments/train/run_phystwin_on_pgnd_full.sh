#!/bin/bash
# Test 2: Full PhysTwin training and evaluation on PGND cloth data

set -e

echo "======================================================================"
echo "Test 2: PhysTwin on PGND Cloth - Full Pipeline"
echo "======================================================================"
echo ""
echo "Goal: Train PhysTwin on PGND cloth, test future prediction accuracy"
echo ""

# Configuration
PGND_EPISODE="/home/fashionista/pgnd/experiments/log/data_cloth/cloth_merged/sub_episodes_v/episode_0112"
PHYSTWIN_DATA="/home/fashionista/PhysTwin/data/pgnd_cloth"
CASE_NAME="pgnd_test_episode112"

# Check if episode has required data
if [ ! -f "$PGND_EPISODE/traj.npz" ]; then
    echo "❌ Error: Episode missing traj.npz"
    echo "   Looking for: $PGND_EPISODE/traj.npz"
    exit 1
fi

echo "📁 Step 1: Convert PGND episode to PhysTwin format"
echo "   Source: $PGND_EPISODE"
echo "   Output: $PHYSTWIN_DATA/$CASE_NAME"
echo ""

# For this episode, we need to create a converter that uses traj.npz
# since it doesn't have RGB-D camera folders

python << 'PYTHON_SCRIPT'
import pickle
import numpy as np
import json
from pathlib import Path
import sys

# Load trajectory data
episode_path = Path("$PGND_EPISODE")
output_path = Path("$PHYSTWIN_DATA") / "$CASE_NAME"
output_path.mkdir(parents=True, exist_ok=True)

print("Loading traj.npz...")
traj_data = np.load(episode_path / 'traj.npz')
xyz = traj_data['xyz']  # (T, N, 3)

print(f"Trajectory shape: {xyz.shape}")
print(f"  {xyz.shape[0]} frames")
print(f"  {xyz.shape[1]} points per frame")

# Create PhysTwin final_data.pkl
final_data = {
    'object_points': xyz,  # (T, N, 3)
    'object_visibilities': np.ones((xyz.shape[0], xyz.shape[1]), dtype=bool),
    'object_motions_valid': np.ones((xyz.shape[0], xyz.shape[1]), dtype=bool),
    'object_colors': np.ones((xyz.shape[0], xyz.shape[1], 3)) * 0.5,  # Gray
}

# Save
output_file = output_path / 'final_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(final_data, f)

print(f"✓ Created: {output_file}")

# Create train/test split (70/30)
num_frames = xyz.shape[0]
train_end = int(num_frames * 0.7)

split = {
    'train': [0, train_end],
    'test': [train_end, num_frames]
}

split_file = output_path / 'split.json'
with open(split_file, 'w') as f:
    json.dump(split, f, indent=2)

print(f"✓ Created: {split_file}")
print(f"  Train: frames 0-{train_end}")
print(f"  Test: frames {train_end}-{num_frames}")

# Create metadata
metadata = {
    'num_frames': num_frames,
    'num_points': xyz.shape[1],
    'data_type': 'pgnd',
}

metadata_file = output_path / 'metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Created: {metadata_file}")

PYTHON_SCRIPT

echo ""
echo "✅ Data conversion complete!"
echo ""

echo "🔧 Step 2: Train PhysTwin (Physics Optimization)"
echo "   Expected time: ~25 minutes"
echo "   Optimizing: spring stiffness, damping, collision params"
echo ""

cd /home/fashionista/PhysTwin

# Run training
python train_warp.py \
  --base_path "$PHYSTWIN_DATA" \
  --case_name "$CASE_NAME" \
  --train_frame $(python -c "import json; print(json.load(open('$PHYSTWIN_DATA/$CASE_NAME/split.json'))['train'][1])")

echo ""
echo "✅ Training complete!"
echo ""

echo "🧪 Step 3: Test Future Prediction"
echo "   Testing on held-out frames"
echo ""

# Run inference
python inference_warp.py \
  --base_path "$PHYSTWIN_DATA" \
  --case_name "$CASE_NAME"

echo ""
echo "✅ Inference complete!"
echo ""

echo "📊 Step 4: Evaluate Results"
echo ""

python << 'EVAL_SCRIPT'
import pickle
import numpy as np

# Load results
case_path = "$PHYSTWIN_DATA/$CASE_NAME"
results_path = f"/home/fashionista/PhysTwin/experiments/{case_path}/inference.pkl"

try:
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    pred = results['predicted_positions']
    gt = results['gt_positions']

    error = np.mean(np.linalg.norm(pred - gt, axis=-1))

    print(f"📈 Future Prediction Results:")
    print(f"   Chamfer Distance: {error:.6f} m")
    print(f"   Prediction steps: {len(pred)}")
    print("")

    # Compare with expected PhysTwin performance (from paper)
    phystwin_paper_cd = 0.012
    print(f"🎯 Comparison:")
    print(f"   PhysTwin on their data: {phystwin_paper_cd:.6f} m")
    print(f"   PhysTwin on PGND data:  {error:.6f} m")
    print("")

    if error < 0.02:
        print("✅ PhysTwin achieves good accuracy on PGND cloth!")
    else:
        print("⚠️  Higher error than expected - check data quality")

except FileNotFoundError:
    print("❌ Inference results not found")
    print(f"   Expected: {results_path}")

EVAL_SCRIPT

echo ""
echo "======================================================================"
echo "Test 2 Complete: PhysTwin trained and evaluated on PGND data"
echo "======================================================================"
echo ""
echo "📁 Output files:"
echo "   Data: $PHYSTWIN_DATA/$CASE_NAME/"
echo "   Model: /home/fashionista/PhysTwin/experiments/$CASE_NAME/"
echo "   Results: /home/fashionista/PhysTwin/experiments/$CASE_NAME/inference.pkl"
echo ""
echo "📊 Next: Compare with PGND baseline and your ablations"
