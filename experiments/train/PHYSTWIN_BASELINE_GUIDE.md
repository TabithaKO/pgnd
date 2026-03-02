# PhysTwin Baseline: Setup and Training Guide

**Purpose:** Add PhysTwin as a physics-based baseline for comparison with PGND
**Repository:** https://github.com/Jianghanxiao/PhysTwin
**Paper:** Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos (ICCV 2025)

## 🔑 Key Insights (Updated After Paper Review)

**PhysTwin's training is NOT purely "self-supervised from images":**

1. **Stage 1** optimizes physics parameters using:
   - Geometric loss (Chamfer distance on 3D point clouds)
   - Motion tracking loss (pseudo-tracking from CoTracker3 vision model)
   - **NO rendering loss** - appearance doesn't affect physics

2. **Stage 2** optimizes Gaussian appearance using:
   - Rendering loss (L1 + D-SSIM image reconstruction)
   - **Physics parameters frozen** - rendering quality doesn't improve physics

**Critical difference from PGND:**
- **PGND**: Ground-truth trajectories → learns dynamics → 40 FPS inference
- **PhysTwin**: Pseudo-tracking from vision → optimizes physics params → 2-5 FPS inference
- **PGND Ablations 1/2**: Ground-truth trajectories + rendering loss → learns dynamics + optimizes appearance

**Why as baseline:** Tests whether physics-based simulation (with noisy pseudo-tracking) can compete with learned dynamics (with perfect trajectory supervision).

---

## Overview

### What is PhysTwin?

PhysTwin reconstructs deformable objects from videos using:
- **Physics-based prior:** Differentiable spring-mass simulation (Nvidia Warp)
- **Vision-based tracking supervision:** CoTracker3 for pseudo-ground-truth particle tracking
- **Two-stage optimization:**
  1. **Stage 1 - Physics & Geometry** (Zero-order CMA-ES + First-order Adam):
     - Optimizes geometry, topology, physical parameters (spring stiffness, control forces)
     - Supervision: `C_geometry` (Chamfer distance) + `C_motion` (tracking error)
     - Does NOT use rendering loss
  2. **Stage 2 - Appearance** (First-order Adam):
     - Optimizes Gaussian appearance parameters only (color, opacity, rotation, scaling)
     - Supervision: `C_render` (L1 + D-SSIM image reconstruction)
     - Physics parameters frozen - rendering loss does NOT affect physics

### Why as a Baseline?

| Aspect | PGND (Ours) | PhysTwin (Baseline) |
|--------|-------------|---------------------|
| **Dynamics Model** | Learned (MLP) | Physics-based (Spring-Mass) |
| **Training Supervision** | Supervised (ground-truth particle trajectories) | Vision-based (pseudo-tracking from CoTracker3 + RGB images) |
| **Requires Depth** | No (RGB-only) | Yes (depth maps for 3D reconstruction) |
| **Inference Speed** | Fast (~40 FPS) | Slow (~2-5 FPS physics sim) |
| **Generalization** | Requires training data | Physics priors transfer |
| **Accuracy** | High (fits training distribution) | Moderate (physics approximation) |

**Key Comparison:** Does learned dynamics (PGND) outperform physics-based simulation (PhysTwin) for cloth manipulation prediction?

### Supervision Signals Explained

**PhysTwin's supervision is NOT purely "self-supervised from images"** - it requires:

1. **Stage 1 - Physics & Geometry Optimization:**
   - **C_geometry**: Chamfer distance between predicted 3D points X̂_t and observed sparse 3D point cloud X_t
     - Point cloud obtained from: segmentation (Grounded-SAM-2) + depth maps + camera calibration
   - **C_motion**: Tracking error between predicted particle positions x̂_i^t and pseudo-ground-truth tracking x_i^t
     - Tracking obtained from: CoTracker3 (vision foundation model) for 2D tracking → lifted to 3D via depth unprojection
   - **Physics parameters optimized**: spring stiffness, damping, control forces
   - **Rendering loss NOT used in this stage**

2. **Stage 2 - Appearance Optimization:**
   - **C_render**: L1 + D-SSIM loss between rendered images Î_{i,t} and ground-truth RGB images I_{i,t}
   - **Only appearance parameters optimized**: Gaussian color, opacity, rotation, scaling at first frame
   - **Physics parameters frozen** - they are NOT updated based on rendering quality
   - Uses LBS (Linear Blend Skinning) to deform Gaussians based on fixed spring-mass dynamics

**Key Insight:** The rendering loss improves visual quality but does NOT improve physics accuracy. Physics parameters are determined solely by geometric/tracking losses in Stage 1.

**Comparison with PGND:**
- **PGND**: Uses ground-truth particle trajectories from simulation → trains learned dynamics model
- **PhysTwin**: Uses pseudo-tracking from vision models → optimizes physics parameters to match pseudo-tracking
- **PGND Ablation 1/2**: Uses ground-truth trajectories → trains learned dynamics + optimizes rendering with frozen/trainable GS

---

## Setup

### 1. Environment

PhysTwin is already cloned at `/home/fashionista/PhysTwin`

```bash
cd /home/fashionista/PhysTwin

# Check current CUDA version
nvcc --version
# Expected: CUDA 12.1+ (you have CUDA compatible with RTX 5070 Ti)

# Create PhysTwin conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install dependencies
bash ./env_install/env_install.sh

# Download pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

**Note:** PhysTwin uses Nvidia Warp for differentiable physics simulation. This may conflict with your existing diff-gaussian-rasterization. Use separate conda environments.

---

### 2. Download PhysTwin Data

PhysTwin provides cloth manipulation datasets that can serve as a standardized benchmark:

```bash
cd /home/fashionista/PhysTwin

# Download processed cloth data (includes 8 cloth cases)
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip
unzip data.zip

# Download pre-optimized material parameters (zero-order stage)
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip
unzip experiments_optimization.zip

# Download trained models (first-order stage)
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip
unzip experiments.zip

# Download Gaussian Splatting reconstructions
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip
unzip gaussian_output.zip
```

**Expected data structure:**
```
/home/fashionista/PhysTwin/
├── data/
│   └── different_types/
│       ├── single_lift_cloth_1/
│       │   ├── color/             # RGB frames
│       │   ├── depth/             # Depth frames
│       │   ├── calibrate.pkl      # Camera extrinsics
│       │   ├── metadata.json      # Camera intrinsics + info
│       │   ├── final_data.pkl     # Processed tracking data
│       │   └── split.json         # Train/test split
│       ├── single_lift_cloth_3/
│       ├── single_lift_cloth_4/
│       ├── double_lift_cloth_1/
│       └── double_lift_cloth_3/
├── experiments_optimization/      # Stage 1 results
├── experiments/                   # Stage 2 results
└── gaussian_output/               # GS reconstructions
```

---

## Training PhysTwin

### Option 1: Train on PhysTwin's Cloth Dataset (Recommended for Baseline)

This provides a standardized comparison - both PGND and PhysTwin trained on the same data.

```bash
cd /home/fashionista/PhysTwin
conda activate phystwin

# Train on all cloth cases (8 cases)
python script_train.py

# Or train on specific cloth case
python train_warp.py \
  --base_path ./data/different_types \
  --case_name single_lift_cloth_1 \
  --train_frame 80  # Number of frames to train on
```

**Cloth cases available:**
- `single_lift_cloth_1`
- `single_lift_cloth_3`
- `single_lift_cloth_4`
- `single_clift_cloth_1`
- `single_clift_cloth_3`
- `double_lift_cloth_1`
- `double_lift_cloth_3`

**Training time:**
- Zero-order (CMA-ES): ~12 minutes per case
- First-order (Adam): ~5 minutes per case
- **Total per case: ~17 minutes** (much faster than PGND's 30 hours!)

---

### Option 2: Adapt Your PGND Data to PhysTwin Format

To compare on your exact dataset, you'll need to convert PGND data format to PhysTwin format.

**Required conversions:**

1. **Camera calibration:**
```python
# PGND format: experiments/log/data_cloth/{recording}/calibration/
#   - intrinsics.npy: (4, 3, 3) intrinsic matrices
#   - rvecs.npy: (4, 3, 1) rotation vectors
#   - tvecs.npy: (4, 3, 1) translation vectors

# PhysTwin format:
#   - calibrate.pkl: list of 4×4 camera-to-world matrices
#   - metadata.json: {"intrinsics": [[...]], "WH": [width, height]}

import pickle
import json
import numpy as np
import cv2

# Load PGND calibration
intrinsics = np.load('calibration/intrinsics.npy')  # (4, 3, 3)
rvecs = np.load('calibration/rvecs.npy')            # (4, 3, 1)
tvecs = np.load('calibration/tvecs.npy')            # (4, 3, 1)

# Convert to camera-to-world matrices
c2ws = []
for i in range(4):
    R = cv2.Rodrigues(rvecs[i])[0]  # (3, 3)
    t = tvecs[i, :, 0]              # (3,)

    # World-to-camera: x_cam = R @ x_world + t
    # Camera-to-world: x_world = R^T @ (x_cam - t) = R^T @ x_cam - R^T @ t
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    c2ws.append(c2w)

# Save calibrate.pkl
with open('calibrate.pkl', 'wb') as f:
    pickle.dump(c2ws, f)

# Save metadata.json
metadata = {
    "intrinsics": intrinsics.tolist(),
    "WH": [848, 480]  # Your image dimensions
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
```

2. **RGB frames:**
```bash
# PGND: experiments/log/data_cloth/{recording}/episode_XXXX/camera_1/rgb/*.jpg
# PhysTwin: data/different_types/{case_name}/color/{frame_id:06d}.jpg

# Create symlinks or copy
mkdir -p phystwin_data/my_cloth_case/color
cp experiments/log/data_cloth/*/episode_*/camera_1/rgb/*.jpg \
   phystwin_data/my_cloth_case/color/
```

3. **Depth frames:**
```bash
# PGND: May not have depth (if using RGB-only GS reconstruction)
# PhysTwin: Requires depth for 3D tracking

# If you have depth from sensors:
mkdir -p phystwin_data/my_cloth_case/depth
# Copy depth frames...

# If you DON'T have depth:
# You'll need to run PhysTwin's data processing pipeline which includes:
#   - Grounded-SAM-2 for segmentation
#   - Trellis for 3D shape estimation
#   - Particle tracking
```

4. **Run PhysTwin data processing:**
```bash
cd /home/fashionista/PhysTwin

# Process your converted data
python process_data.py \
  --base_path ./phystwin_data \
  --case_name my_cloth_case
```

**Note:** This is complex! For a fair baseline comparison, **Option 1 (train on PhysTwin's data) is recommended**.

---

## Evaluation

### Metrics

PhysTwin and PGND should be compared on:

1. **Chamfer Distance:** 3D particle position accuracy
2. **Image Quality:** PSNR, SSIM on rendered images
3. **Inference Speed:** FPS for rollout prediction
4. **Training Time:** Hours to converge

### Run PhysTwin Inference

```bash
cd /home/fashionista/PhysTwin
conda activate phystwin

# Render predictions on test frames
python inference_warp.py \
  --base_path ./data/different_types \
  --case_name single_lift_cloth_1

# Evaluate Chamfer distance
python evaluate_chamfer.py \
  --case_name single_lift_cloth_1

# Render for visual comparison
bash gs_run_simulate.sh
python visualize_render_results.py
```

### Compare Against PGND

Create a comparison script:

```python
# Compare PhysTwin vs PGND on cloth manipulation

import numpy as np
import json

def load_phystwin_results(case_name):
    """Load PhysTwin predictions."""
    path = f"/home/fashionista/PhysTwin/experiments/{case_name}/pred_pos.npy"
    return np.load(path)

def load_pgnd_results(episode_name):
    """Load PGND predictions."""
    path = f"/home/fashionista/pgnd/experiments/log/eval/{episode_name}/state/episode_0000.pt"
    import torch
    data = torch.load(path)
    return data['x'].cpu().numpy()

def compute_chamfer(pred, gt):
    """Compute Chamfer distance."""
    # pred: (T, N, 3)
    # gt: (T, N, 3)
    from scipy.spatial.distance import cdist

    chamfers = []
    for t in range(len(pred)):
        # Forward: pred -> gt
        dist_matrix = cdist(pred[t], gt[t])
        forward = dist_matrix.min(axis=1).mean()

        # Backward: gt -> pred
        backward = dist_matrix.min(axis=0).mean()

        chamfers.append((forward + backward) / 2)

    return np.array(chamfers)

# Example comparison
phystwin_pred = load_phystwin_results('single_lift_cloth_1')
pgnd_pred = load_pgnd_results('episode_0000')

chamfer = compute_chamfer(phystwin_pred, gt_particles)
print(f"PhysTwin Chamfer: {chamfer.mean():.6f}")
```

---

## Integration with Your Ablation Framework

Add PhysTwin to your ablation guide as Baseline B:

### Updated Ablation Roadmap

| Method | Dynamics Model | Render Loss | Training Supervision | Status |
|--------|---------------|-------------|---------------------|--------|
| **Baseline A (PGND)** | Learned (MLP) | ❌ No | Ground-truth particle trajectories | ✅ Complete |
| **Baseline B (PhysTwin)** | Physics (Spring-Mass) | ❌ No (decoupled) | Pseudo-tracking (CoTracker3) + RGB-D | 📋 Next |
| **Ablation 1a** | Learned (MLP) | ✅ Frozen GS + SSIM | Ground-truth trajectories + RGB | ⏳ Running |
| **Ablation 1b** | Learned (MLP) | ✅ Frozen GS + DINOv2 | Ground-truth trajectories + RGB | ⏳ Running |
| **Ablation 2a** | Learned (MLP) | ✅ Trainable GS + SSIM | Ground-truth trajectories + RGB | 📋 Planned |
| **Ablation 2b** | Learned (MLP) | ✅ Trainable GS + DINOv2 | Ground-truth trajectories + RGB | 📋 Planned |
| **Ablation 2c** | Learned (MLP) | ✅ Trainable GS + Combined | Ground-truth trajectories + RGB | 📋 Planned |

### Key Comparisons

**Baseline A vs Baseline B:**
- **Question**: Learned dynamics vs physics-based dynamics?
- **PGND**: Ground-truth trajectories → learns dynamics, fast inference (40 FPS)
- **PhysTwin**: Pseudo-tracking from vision → optimizes physics params, slow inference (2-5 FPS)
- **Key difference**: PGND has perfect supervision; PhysTwin has noisy pseudo-tracking

**Baseline A vs Ablation 1:**
- **Question**: Does adding frozen GS render loss improve learned dynamics?
- **Baseline A**: Only trajectory loss
- **Ablation 1**: Trajectory loss + frozen GS render loss (SSIM or DINOv2)
- **Hypothesis**: Render loss provides additional signal but may not improve dynamics much (GS frozen)

**Baseline B vs Ablations 1/2:**
- **Question**: Physics-based vs learned dynamics (with render loss)?
- **PhysTwin**: Physics approximation, decoupled appearance optimization
- **Ablations 1/2**: Learned dynamics with coupled rendering optimization
- **Hypothesis**: Learned should be more accurate on in-distribution; physics may generalize better

**Ablation 1 vs Ablation 2:**
- **Question**: Frozen vs trainable GS parameters (your main contribution)?
- **Ablation 1**: Frozen GS, LBS deformation, single backward pass
- **Ablation 2**: Trainable GS, mesh anchoring, dual backward passes
- **Hypothesis**: Trainable GS allows appearance to adapt to dynamics, improving both rendering and dynamics accuracy

---

## Expected Results

### PhysTwin Strengths
- ✅ **Physics interpretability:** Material parameters have physical meaning
- ✅ **Zero-shot transfer:** Physics priors generalize to new scenarios
- ✅ **Fast training:** ~17 min vs PGND's 30 hours
- ✅ **No ground-truth trajectories:** Uses pseudo-tracking from vision models (CoTracker3)
- ✅ **Works on real videos:** Can process real-world RGB-D recordings

### PhysTwin Weaknesses
- ❌ **Slower inference:** 2-5 FPS vs PGND's 40 FPS
- ❌ **Physics approximation:** Spring-mass may not capture all cloth dynamics
- ❌ **Requires depth maps:** Needs depth sensor or monocular depth estimation
- ❌ **Requires vision models:** Depends on quality of CoTracker3 pseudo-tracking
- ❌ **Local minima:** Optimization-based, can get stuck
- ❌ **Decoupled optimization:** Rendering quality doesn't improve physics accuracy

### Expected Performance Ranking (Chamfer Distance, lower = better)

1. **PGND Baseline (no render loss):** ~0.005 (trained on exact trajectories)
2. **Ablation 2c (trainable GS + combined loss):** ~0.006 (render loss improves)
3. **Ablation 1b (frozen GS + DINOv2):** ~0.007
4. **PhysTwin:** ~0.015 (physics approximation error)

**Insight:** Learned dynamics should outperform physics-based simulation on in-distribution data, but PhysTwin may generalize better to out-of-distribution scenarios.

---

## Quick Start Commands

```bash
# 1. Setup PhysTwin environment
cd /home/fashionista/PhysTwin
conda create -y -n phystwin python=3.10
conda activate phystwin
bash ./env_install/env_install.sh
bash ./env_install/download_pretrained_models.sh

# 2. Download cloth data
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip
unzip data.zip
wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip
unzip experiments_optimization.zip

# 3. Train on cloth cases (fast! ~17 min per case)
python script_train.py

# 4. Evaluate
python evaluate_chamfer.py --case_name single_lift_cloth_1

# 5. Visualize
python interactive_playground.py --case_name single_lift_cloth_1 --n_ctrl_parts 1
```

---

## Troubleshooting

### Issue: CUDA version mismatch

PhysTwin requires CUDA 12.1+. Check with:
```bash
nvcc --version
```

If incompatible, follow RTX 5090 setup instructions in README.

### Issue: Warp installation fails

```bash
# Install Warp separately
pip install warp-lang
```

### Issue: Gaussian Splatting conflict

PhysTwin and PGND both use diff-gaussian-rasterization but may have different versions. Use separate conda environments:
- `pgnd` environment for PGND experiments
- `phystwin` environment for PhysTwin baseline

### Issue: Missing depth data for your dataset

PhysTwin requires depth. Options:
1. Use PhysTwin's provided cloth dataset (has depth)
2. Use monocular depth estimation (ZoeDepth, MiDaS)
3. Modify PhysTwin to work RGB-only (requires code changes)

---

## Additional Resources

- **PhysTwin Paper:** https://jianghanxiao.github.io/phystwin-web/phystwin.pdf
- **PhysTwin Code:** https://github.com/Jianghanxiao/PhysTwin
- **Interactive Demo:** See PhysTwin README for interactive playground

---

## Contact

For PhysTwin-specific questions, contact the authors: hanxiao.jiang@columbia.edu

For integration with your PGND experiments, this guide should cover the basics!
