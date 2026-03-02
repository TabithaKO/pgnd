# PGND Ablation Studies: Unified Training Guide

**Author:** PhD Research - Cloth Manipulation with Robotic Arms
**Date:** February 11, 2026
**System:** PGND + Differentiable Rendering

---

## Table of Contents

1. [Overview of All Ablations](#overview-of-all-ablations)
2. [Ablation 2a/2b/2c: Trainable Mesh-Constrained GS](#ablation-2abc-trainable-mesh-constrained-gs)
3. [Future Ablations](#future-ablations)
4. [Prerequisites](#prerequisites)
5. [Data Requirements](#data-requirements)
6. [Configuration](#configuration)
7. [Training Procedure](#training-procedure)
8. [Monitoring](#monitoring)
9. [Expected Outputs](#expected-outputs)
10. [Troubleshooting](#troubleshooting)

---

## Overview of All Ablations

### Ablation Study Roadmap

| Ablation | Research Question | Status | Training Time |
|----------|------------------|--------|---------------|
| **Ablation 0** | Baseline PGND (no render loss) | ✅ Complete | ~30h |
| **Ablation 1a** | Frozen GS + LBS + SSIM loss | ⏳ Running | ~30h |
| **Ablation 1b** | Frozen GS + LBS + DINOv2 loss | ⏳ Running | ~36h |
| **Ablation 2a** | Trainable mesh-GS + SSIM loss | 📋 Next | ~30h |
| **Ablation 2b** | Trainable mesh-GS + DINOv2 loss | 📋 Planned | ~36h |
| **Ablation 2c** | Trainable mesh-GS + SSIM + DINOv2 | 📋 Planned | ~38h |
| **Ablation 3** | BPA vs Poisson mesh construction | 💡 Future | ~30-32h |
| **Ablation 4** | Point cloud dropout robustness | 💡 Future | ~30h |

### Key Comparisons

**Ablation 1 vs 2:** Frozen GS (LBS deformation) vs Trainable GS (mesh-constrained barycentric)
- Tests whether trainable GS parameters improve dynamics learning
- Compares gradient flow through LBS vs barycentric interpolation

**Ablation 2a vs 2b vs 2c:** SSIM vs DINOv2 vs Combined losses
- Tests which image loss provides best supervision signal
- Compares structural (SSIM) vs semantic (DINOv2) similarity

**Ablation 3:** BPA vs Poisson mesh construction
- Tests 1:1 vertex mapping vs interpolated watertight mesh
- Evaluates gradient flow quality and reconstruction accuracy

**Ablation 4:** Robustness to sensor noise
- Tests model performance with incomplete/noisy point clouds
- Evaluates generalization to real-world sensor failures

---

## Ablation 2a/2b/2c: Trainable Mesh-Constrained GS

### Goal
Evaluate whether **trainable mesh-constrained Gaussian Splatting** improves dynamics prediction compared to frozen GS (Ablation 1).

### Key Innovation
Instead of freezing GS parameters and using LBS deformation (Ablation 1), Ablation 2:
- **Anchors Gaussians to mesh faces** via barycentric coordinates
- **Trains GS parameters** (colors, scales, rotations, opacities) jointly with dynamics
- **Uses dual backward passes** to separate dynamics and rendering optimization

### Research Question
*Does trainable mesh-constrained GS provide stronger supervision signal than frozen LBS-deformed GS?*

### Variants

**Ablation 2a: SSIM Loss**
- Image loss: L1 + λ_ssim × (1 - SSIM)
- Captures structural similarity
- Baseline for image-space supervision

**Ablation 2b: DINOv2 Loss**
- Image loss: Cosine distance between DINOv2 patch features
- Captures semantic/perceptual similarity
- Robust to lighting and texture variations

**Ablation 2c: Combined Loss**
- Image loss: (1 - λ_dino) × SSIM_loss + λ_dino × DINOv2_loss
- Best of both worlds: structural + semantic
- Expected best performance

---

## Architecture

### Mesh-Constrained Gaussian Model

```
┌─────────────────────────────────────────────────────────────┐
│                    PGND Dynamics Model (fθ)                  │
│                  Predicts: x_t → x_{t+1}                     │
└────────────────┬────────────────────────────────────────────┘
                 │ particles (N, 3)
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              Mesh Construction (BPA or Poisson)             │
│           Converts particles → triangle mesh                │
└────────────────┬────────────────────────────────────────────┘
                 │ mesh vertices, faces
                 ↓
┌─────────────────────────────────────────────────────────────┐
│          Barycentric Interpolation (Differentiable)         │
│   Each Gaussian anchored to face: pos = Σ(b_i * v_i)       │
│   Trainable: barycentric weights (b_0, b_1, b_2)           │
└────────────────┬────────────────────────────────────────────┘
                 │ deformed GS positions
                 ↓
┌─────────────────────────────────────────────────────────────┐
│       Trainable GS Parameters (colors, scales, etc.)        │
│           Jointly optimized with dynamics model             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              Gaussian Rasterizer (Differentiable)           │
│                 Renders RGB image from GS                   │
└────────────────┬────────────────────────────────────────────┘
                 │ rendered image
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              Image Loss (L1 + SSIM or DINOv2)               │
│          Compares rendered vs ground-truth images           │
└─────────────────────────────────────────────────────────────┘
```

### Dual Backward Pass Architecture

```python
# Forward: Dynamics + Rendering
x_pred = dynamics_model(x_t, v_t)                   # Predict next state
mesh = build_mesh(x_pred)                            # Construct mesh
gs_positions = barycentric_interp(mesh, bary_coords) # Deform GS
rendered_image = rasterize(gs_positions, gs_params)  # Render
loss_image = image_loss(rendered_image, gt_image)    # Compare

# Backward Pass 1: Dynamics (receives both geometry + render loss)
loss_dynamics = loss_x + loss_v + λ_render * loss_image
loss_dynamics.backward()
dynamics_optimizer.step()

# Backward Pass 2: GS Parameters (receives only render loss)
gs_optimizer.zero_grad()
loss_gs = loss_image  # Unweighted
loss_gs.backward()
gs_optimizer.step()
```

**Critical Design:** The render loss is weighted by `λ_render` for dynamics but unweighted for GS parameters. This prevents over-suppression of GS gradients.

---

## Comparison: Ablation 1 vs Ablation 2

| Aspect | Ablation 1 (Frozen GS + LBS) | Ablation 2 (Trainable Mesh-GS) |
|--------|------------------------------|--------------------------------|
| **GS Parameters** | Frozen (requires_grad=False) | Trainable (5 param groups) |
| **Deformation Method** | LBS (K-NN inverse distance) | Barycentric interpolation |
| **Mesh Construction** | None (direct LBS from particles) | BPA or Poisson reconstruction |
| **Coordinate System** | Particles in preprocessed space | Mesh vertices = particles (1:1) |
| **Backward Passes** | Single (dynamics only) | Dual (dynamics + GS) |
| **Optimizers** | 1 (Adam for dynamics) | 2 (Adam for dynamics + Adam for GS) |
| **Memory Usage** | ~4.2 GB (actual) | ~10-14 GB (estimated) |
| **Training Time** | ~30 hours (100k iters) | ~30 hours (100k iters, expected) |
| **Rotation Estimation** | Procrustes (SVD) for LBS | Rigid point registration (roma) |

### When to Use Each

- **Ablation 1:** When you want fast training, lower memory, and GS reconstruction quality is already excellent
- **Ablation 2:** When GS reconstruction has artifacts OR you expect cloth geometry to change significantly during training

---

## Prerequisites

### 1. System Requirements

```bash
# GPU
- NVIDIA GPU with ≥16 GB VRAM (RTX 5070 Ti or better)
- CUDA 11.8+

# Python Environment
- Python 3.8+
- PyTorch 2.0+
- diff-gaussian-rasterization (from third-party/)
- roma (for rotation estimation)
- torch_geometric (for mesh operations)
- scipy (for mesh construction)
```

### 2. Installation Check

```bash
# Verify all dependencies
python -c "import torch; import roma; import torch_geometric; print('✓ All imports OK')"

# Test mesh construction
python experiments/train/build_cloth_mesh_bpa.py \
  --episode_dir experiments/log/data_cloth/cloth_merged/sub_episodes_v/episode_0162 \
  --output_dir /tmp/test_mesh

# Test Gaussian anchoring
python experiments/train/anchor_gaussians_to_mesh.py \
  --mesh_npz /tmp/test_mesh/mesh.npz \
  --splat_path experiments/log/data_cloth/1223_cloth_gello_cali2_processed/episode_0002/gs/000000.splat \
  --output_npz /tmp/test_anchors.npz
```

### 3. Data Preparation

Run the episode matching script to ensure you have .splat files for your training episodes:

```bash
python experiments/train/find_matching_episodes.py \
  --cloth_merged experiments/log/data_cloth/cloth_merged \
  --data_cloth experiments/log/data_cloth \
  --output episode_matches.json

# Expected output:
# Found 286 episodes with matching .splat files (out of 650 total)
```

**Important:** Only episodes with `.splat` files can use render loss. Episodes without GS data will skip render loss and train on geometry loss only.

---

## Data Requirements

### Directory Structure

```
experiments/log/
├── data_cloth/                          # Source data with GS reconstructions
│   ├── 1223_cloth_gello_cali2_processed/
│   │   ├── episode_0002/
│   │   │   ├── gs/                      # ← .splat files here
│   │   │   │   ├── 000000.splat
│   │   │   │   ├── 000001.splat
│   │   │   │   └── ...
│   │   │   ├── calibration/             # ← Camera params here
│   │   │   │   ├── intrinsics.npy
│   │   │   │   ├── rvecs.npy
│   │   │   │   └── tvecs.npy
│   │   │   └── camera_1/                # ← RGB frames here
│   │   │       ├── rgb/
│   │   │       │   ├── 000000.jpg
│   │   │       │   └── ...
│   │   │       └── mask/
│   │   │           ├── 000000.png
│   │   │           └── ...
│   │   └── ...
│   └── cloth_merged/                    # Preprocessed training data
│       ├── sub_episodes_v/
│       │   ├── episode_0000/
│       │   │   ├── traj.npz             # ← Particle trajectories
│       │   │   └── meta.txt             # ← Maps to source episode
│       │   └── ...
│       └── metadata.json                # ← Episode → source mapping
└── gs_checkpoints/                      # Output: trained GS models (per episode)
    └── ...
```

### Episode Matching Logic

Each preprocessed episode (`cloth_merged/sub_episodes_v/episode_XXXX/`) has:
- `traj.npz`: Particle trajectories in PGND preprocessed space [0,1]³
- `meta.txt`: Contains `[source_episode_id, source_frame_start, num_frames]`

The training script:
1. Reads `meta.txt` to find source episode
2. Loads `.splat` file closest to `source_frame_start`
3. Transforms GS positions from world coordinates → preprocessed space
4. Constructs mesh and anchors Gaussians

---

## Configuration

### Base Configuration File

Create `experiments/configs/ablation2a.yaml`:

```yaml
# experiments/configs/ablation2a.yaml
# Ablation 2a: SSIM Loss Only

defaults:
  - default  # Inherit base PGND config

train:
  name: 'ablation2a_mesh_gs_ssim'

  # Training schedule (same as Ablation 1)
  num_iterations: 100000
  batch_size: 1                    # Reduce to 1 if memory constrained
  training_start_episode: 162
  training_end_episode: 242

  # Enable render loss
  use_render_loss: true

  # Render loss weights
  lambda_render: 0.1               # Weight in dynamics backward pass
  lambda_ssim: 0.2                 # SSIM vs L1 weight (0.2 = 20% SSIM, 80% L1)
  lambda_dino: 0.0                 # DINOv2 feature loss (disabled for 2a)

  # Rendering configuration
  render_every_n_steps: 2          # Render every N rollout steps
  render_camera_id: 1              # Camera ID (0-3)

  # Mesh method
  mesh_method: 'bpa'               # 'bpa' (1:1 vertex mapping) or 'poisson'

  # GS parameter filtering
  gs_opacity_threshold: 0.1        # Filter Gaussians with opacity < 0.1

  # GS optimizer learning rates
  gs_lr_position: 1.0e-4           # Barycentric coordinates LR
  gs_lr_color: 2.5e-3              # RGB colors LR
  gs_lr_scale: 5.0e-3              # Scales LR
  gs_lr_rotation: 1.0e-3           # Rotations LR
  gs_lr_opacity: 5.0e-2            # Opacities LR

  # Dynamics model optimizer (same as baseline)
  material_lr: 1.0e-4
  material_grad_max_norm: 10.0
```

### Mesh Method Selection

**Ball Pivoting Algorithm (BPA):** *(Recommended)*
- **Pros:** Preserves 1:1 vertex-particle correspondence, fast, no interpolation error
- **Cons:** May create holes in sparse regions, mesh is coarser
- **Use when:** Particle distribution is dense enough to form continuous surface

**Poisson Surface Reconstruction:**
- **Pros:** Produces smooth, watertight mesh even from sparse points
- **Cons:** Creates new vertices (requires RBF interpolation), slower, adds complexity
- **Use when:** Particles are sparse OR you need fine surface detail

---

## Training Procedure

### Step 1: Verify Data Availability

```bash
# Check that your training episodes have .splat files
python experiments/train/find_matching_episodes.py \
  --cloth_merged experiments/log/data_cloth/cloth_merged \
  --data_cloth experiments/log/data_cloth \
  --output episode_matches.json

# Verify episodes 162-242 have matches
cat episode_matches.json | jq '.[] | select(.episode_idx >= 162 and .episode_idx <= 242) | .episode_name'
```

### Step 2: Dry Run (Synthetic Test)

Test the mesh-constrained GS architecture with synthetic data:

```bash
python experiments/train/test_mesh_gaussian_model_synthetic.py

# Expected output:
# TEST 1: Manual Initialization ✓
# TEST 2: Forward Pass ✓
# TEST 3: Mesh Deformation ✓
# TEST 4: Gradient Flow ✓
# TEST 5: Optimizer Setup ✓
# TEST 6: Save/Load ✓
# ALL TESTS PASSED!
```

### Step 3: Real Data Test (Single Episode)

Test with actual PGND episode + .splat data:

```bash
python experiments/train/test_mesh_gaussian_model.py \
  --episode_dir experiments/log/data_cloth/cloth_merged/sub_episodes_v/episode_0162 \
  --splat_path experiments/log/data_cloth/1223_cloth_gello_cali2_processed/episode_0002/gs/000000.splat \
  --mesh_method bpa

# Expected output:
# Loaded episode: 1627 particles
# Loaded GS: 8341 Gaussians (after opacity filter)
# Mesh: 1627 vertices, 3230 faces
# Anchoring: mean distance to face = 0.46
# TEST 1-6: ✓ All passed
```

**Note:** Mean distance to face may be >0.3 due to coordinate space mismatch. This is expected and will be corrected during training via `PGNDCoordinateTransform`.

### Step 4: Launch Training

```bash
# Ablation 2a: SSIM Loss Only
python experiments/train/train_eval_ablation2.py \
  gpus=[0] \
  --config-name=ablation2a \
  overwrite=False \
  resume=False \
  debug=False \
  train.name='ablation2a_mesh_gs_ssim' \
  train.use_render_loss=true \
  train.lambda_render=0.1 \
  train.lambda_ssim=0.2 \
  train.lambda_dino=0.0 \
  train.mesh_method=bpa \
  train.training_start_episode=162 \
  train.training_end_episode=242 \
  train.batch_size=1 \
  train.num_iterations=100000

# Training will take ~30 hours for 100k iterations
```

### Step 5: Resume Training (if interrupted)

```bash
python experiments/train/train_eval_ablation2.py \
  gpus=[0] \
  --config-name=ablation2a \
  resume=True \
  train.resume_iteration=50000  # Last saved checkpoint
```

---

## Monitoring

### Real-Time GPU Monitoring

```bash
# Terminal 1: Training
python experiments/train/train_eval_ablation2.py ...

# Terminal 2: GPU monitor (5s refresh)
watch -n 5 nvidia-smi

# Expected GPU usage:
# - Memory: 10-14 GB
# - Utilization: 80-100%
# - Temperature: <85°C
```

### Weights & Biases (Recommended)

W&B automatically logs:

**Dynamics Losses:**
- `main/loss_x`: Position MSE
- `main/loss_v`: Velocity MSE
- `main/loss_render`: Weighted render loss (λ_render × loss_image)

**GS Losses:**
- `gs/loss_gs`: Unweighted render loss for GS parameters
- `gs/loss_image_l1`: L1 component
- `gs/loss_image_ssim`: SSIM component (if enabled)
- `gs/loss_dino`: DINOv2 component (if enabled)

**Gradient Norms:**
- `stat/material_grad_norm`: Dynamics model gradients
- `stat/gs_grad_norm`: GS parameter gradients

**Visualizations:**
- `render_debug/comparison`: Side-by-side rendered | GT masked | diff images (saved every 100 render calls)

**GS Parameter Statistics:**
- `gs/opacity_mean`, `gs/opacity_std`: Opacity distribution
- `gs/scale_mean`, `gs/scale_std`: Scale distribution
- `gs/color_mean`, `gs/color_std`: RGB color distribution

### Debug Images

Debug images are saved to:
```
experiments/log/ablation2a_mesh_gs_ssim/render_loss_ablation2_debug/
├── debug_000000_step0.jpg
├── debug_000100_step2.jpg
└── ...
```

Each image shows: `Rendered (raw) | GT (masked) | Diff (5x)`

### GS Checkpoints

Per-episode GS models are saved to:
```
experiments/log/gs_checkpoints/
├── episode_0162/
│   ├── mesh.npz              # Mesh vertices, faces
│   ├── gaussian_params.npz   # colors, scales, rotations, opacities
│   └── anchoring.npz         # face_ids, bary_coords
└── ...
```

These checkpoints are saved:
- **Every 10 episodes** during training
- **At the end of training** for all episodes

---

## Expected Outputs

### Training Logs (Console)

```
[ablation2a_mesh_gs_ssim, 14:32:15, iteration    100/100000]
pred.norm 0.0234, e-lr 1.00e-04, e-|grad| 2.156,
loss_x 0.00012345, loss_v 0.00034567, loss_render 0.00000789,
loss_gs 0.00007890

[render_loss] Episode episode_0162: GS setup successful
  Loaded 8341 Gaussians (after opacity filter)
  Mesh: 1627 vertices, 3230 faces (BPA)
  Anchoring: mean distance = 0.112

[render_loss] Render at step 2: loss_image=0.0234 (L1=0.0187, SSIM=0.0047)
```

### Convergence Metrics

**Geometry Losses (MSE):**
- `loss_x`: Should decrease from ~1e-3 to ~1e-4 over 100k iterations
- `loss_v`: Should decrease from ~1e-3 to ~1e-4 over 100k iterations

**Render Losses:**
- `loss_render` (dynamics): Should decrease from ~1e-3 to ~1e-4
- `loss_gs` (GS params): Should decrease from ~1e-2 to ~1e-3

**Gradient Norms:**
- `material_grad_norm`: Should stabilize around 2-5 after warmup
- `gs_grad_norm`: Should stabilize around 5-10 after warmup

**GS Parameter Evolution:**
- Opacities: Should concentrate around 0.8-1.0 for cloth-covered regions
- Scales: Should decrease slightly as GS parameters refine
- Colors: Should match cloth texture (may shift slightly)

### Final Model

After training, you'll have:

```
experiments/log/ablation2a_mesh_gs_ssim/
├── ckpt/
│   ├── 000000.pt              # Initial dynamics checkpoint
│   ├── 010000.pt              # Every 10k iterations
│   └── 100000.pt              # Final dynamics checkpoint
├── gs_checkpoints/
│   ├── episode_0162/          # GS model per episode
│   │   ├── mesh.npz
│   │   ├── gaussian_params.npz
│   │   └── anchoring.npz
│   └── ...
├── render_loss_ablation2_debug/
│   └── debug_*.jpg            # Debug visualizations
├── wandb/                     # W&B logs
└── hydra.yaml                 # Full config
```

---

## Troubleshooting

### Issue 1: OOM (Out of Memory)

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size:**
   ```yaml
   train:
     batch_size: 1  # Down from 2
   ```

2. **Reduce render frequency:**
   ```yaml
   train:
     render_every_n_steps: 4  # Up from 2
   ```

3. **Filter more Gaussians:**
   ```yaml
   train:
     gs_opacity_threshold: 0.2  # Up from 0.1
   ```

4. **Use BPA instead of Poisson:**
   ```yaml
   train:
     mesh_method: 'bpa'  # BPA is more memory-efficient
   ```

### Issue 2: Poor Anchoring Quality

**Symptom:**
```
[render_loss] Anchoring: mean distance to face = 0.45
```

**Expected:** <0.1 for good anchoring
**Diagnosis:** Coordinate space mismatch between .splat (world coords) and mesh (preprocessed coords)

**Solution:**
This is expected initially! The `PGNDCoordinateTransform` in `render_loss_ablation2.py` handles the transform:
- Forward: world → preprocessed (for anchoring)
- Inverse: preprocessed → world (for rendering)

During training, the mesh will align with GS positions as dynamics model learns.

### Issue 3: NaN Gradients

**Symptom:**
```
RuntimeError: Function 'MulBackward0' returned nan gradients
```

**Diagnosis:** Numerical instability in barycentric interpolation or rotation estimation

**Solutions:**
1. **Check barycentric normalization:**
   ```python
   # In mesh_gaussian_model.py line 142
   norm_bary = self.face_bary / (self.face_bary.sum(dim=1, keepdim=True) + 1e-8)
   ```
   Increase epsilon if needed: `1e-6` → `1e-5`

2. **Disable rotation estimation:**
   ```python
   # In mesh_gaussian_model.py line 176-177
   if not HAS_ROMA or deformed_vertices is None:
       return base_rotation
   ```
   Set `HAS_ROMA = False` temporarily

3. **Check for degenerate triangles:**
   ```bash
   # In build_cloth_mesh_bpa.py, add validation
   areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
   assert (areas > 1e-6).all(), "Degenerate triangles detected"
   ```

### Issue 4: GS Gradients Too Small

**Symptom:**
```
stat/gs_grad_norm: 0.0001  # Should be ~5-10
```

**Diagnosis:** GS loss is too weak or GS parameters are saturated

**Solutions:**
1. **Increase render loss weight for GS:**
   Note that `loss_gs` is NOT weighted by `lambda_render`. If gradients are too small, the issue may be:
   - Learning rates too low
   - Image loss too small (check `gs/loss_image_l1`)

2. **Increase GS learning rates:**
   ```yaml
   train:
     gs_lr_position: 2.0e-4  # Up from 1e-4
     gs_lr_color: 5.0e-3     # Up from 2.5e-3
   ```

3. **Check opacity saturation:**
   If all opacities → 1.0, gradients vanish due to sigmoid saturation.
   ```python
   # Lower opacity initialization (in mesh_gaussian_model.py)
   self._opacities = nn.Parameter(
       inverse_sigmoid(gs_opacities_torch * 0.8)  # Scale down initial values
   )
   ```

### Issue 5: Rendering Artifacts

**Symptom:**
Black holes, flickering, or distorted cloth in rendered images

**Diagnosis:**
- Gaussian positions outside camera frustum
- Scales too large/small
- Opacity too low

**Solutions:**
1. **Verify coordinate transform:**
   Check `PGNDCoordinateTransform.inverse_transform()` is correct:
   ```python
   # In render_loss_ablation2.py line 89-93
   positions = (positions_preprocessed - self.translation) / self.scale
   positions = positions @ self.R.T
   ```

2. **Clamp scales during training:**
   ```python
   # In mesh_gaussian_model.py line 247-248
   scales = torch.exp(self._scales)
   scales = torch.clamp(scales, min=1e-4, max=0.1)  # Add clamping
   ```

3. **Increase opacity threshold:**
   ```yaml
   train:
     gs_opacity_threshold: 0.2  # Only keep high-opacity Gaussians
   ```

### Issue 6: Training Too Slow

**Symptom:**
Training takes >40 hours for 100k iterations

**Diagnosis:**
- Rendering too frequently
- Mesh construction too slow (Poisson)
- DINOv2 feature extraction overhead

**Solutions:**
1. **Reduce render frequency:**
   ```yaml
   train:
     render_every_n_steps: 4  # Up from 2
   ```

2. **Use BPA instead of Poisson:**
   ```yaml
   train:
     mesh_method: 'bpa'  # ~10x faster than Poisson
   ```

3. **Disable DINOv2:**
   ```yaml
   train:
     lambda_dino: 0.0  # DINOv2 adds ~20% overhead
   ```

4. **Profile with PyTorch profiler:**
   ```python
   # Add to train_eval_ablation2.py
   with torch.profiler.profile(...) as prof:
       # Training loop
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

---

## Future Ablations

### Ablation 3: Mesh Construction Method Comparison

**Research Question:** Does 1:1 vertex-particle mapping (BPA) provide better gradient flow than interpolated watertight mesh (Poisson)?

**Experiments:**
- **Ablation 3a:** BPA mesh construction (1:1 mapping, may have holes)
- **Ablation 3b:** Poisson reconstruction (watertight, RBF interpolation)

**Key Differences:**
- BPA preserves direct particle-vertex correspondence (faster, simpler gradients)
- Poisson creates smoother surface but requires K-NN interpolation (slower, additional learned mapping)

**Status:** Future work (after Ablation 2 completes)

---

### Ablation 4: Point Cloud Dropout Robustness

**Research Question:** How robust is the model to incomplete/noisy sensor data?

**Experiments:**
- **Ablation 4a:** Full density baseline (0% dropout)
- **Ablation 4b:** Realistic sensor noise (20% dropout)
- **Ablation 4c:** Severe degradation (50% dropout)

**Dropout Methods:**
- Uniform random: Simulate sensor noise across entire cloud
- Spatial/regional: Simulate local occlusions (robot arm blocking view)
- Sensor-specific: Simulate camera failure (drop entire view)

**Status:** Future work (tests real-world deployment robustness)

---

## Ablation 2 Details

### Loss Function Variants

**Ablation 2a: SSIM Loss**

**Configuration:**
```yaml
train:
  lambda_ssim: 0.2
  lambda_dino: 0.0
  mesh_method: 'bpa'  # Use BPA for all Ablation 2 experiments
```

**Rationale:** SSIM captures structural similarity, good baseline for image-space loss

**Expected Results:** Better than Ablation 1 if GS reconstruction has structural errors

---

**Ablation 2b: DINOv2 Feature Loss**

**Configuration:**
```yaml
train:
  lambda_ssim: 0.0
  lambda_dino: 1.0
  mesh_method: 'bpa'
```

**Rationale:** DINOv2 features capture high-level semantic similarity, robust to lighting/texture variations

**Expected Results:** Better generalization if training data has lighting variation

**Training Time:** ~20% slower due to DINOv2 forward pass

---

**Ablation 2c: SSIM + DINOv2 Combined**

**Configuration:**
```yaml
train:
  lambda_ssim: 0.2
  lambda_dino: 0.5
  mesh_method: 'bpa'
```

**Rationale:** Combine structural (SSIM) and semantic (DINOv2) losses

**Expected Results:** Best of both worlds - structural accuracy + semantic robustness

**Training Time:** ~25% slower due to DINOv2 overhead

---

### Mesh Construction (Fixed for Ablation 2)

**Ball Pivoting Algorithm (BPA)** is used for all Ablation 2 experiments:

**Properties:**
- **Preserves 1:1 vertex-particle mapping** (mesh vertex i = particle i)
- No interpolation required (direct correspondence)
- Fast (~0.1s per mesh construction)
- Simple gradient flow (no RBF interpolation layer)

**Why BPA for Ablation 2?**
- Simplest mesh construction method (controls for mesh complexity)
- Direct comparison to Ablation 1 (LBS also uses particles directly)
- Mesh method ablation (BPA vs Poisson) will be studied separately in Ablation 3

**Note:** All experiments use **full-density fused multi-view point clouds** from 4 calibrated cameras. Point cloud dropout ablation will be studied separately in Ablation 4

---

## Comparison Commands

### Compare Ablation 2 Variants (2a vs 2b vs 2c)

```bash
# Compare loss functions: SSIM vs DINOv2 vs Combined
python experiments/train/compare_ablations.py \
  --runs "ablation2a,ablation2b,ablation2c" \
  --output_dir experiments/log/comparison_ablation2 \
  --title "Ablation 2 Loss Function Comparison"
```

### Compare All Ablations (0, 1, 2)

```bash
# Compare all ablation studies
python experiments/train/compare_ablations.py \
  --ablation0_dir experiments/log/baseline_pgnd_no_render_loss \
  --ablation1a_dir experiments/log/ablation1a_frozen_gs_ssim \
  --ablation1b_dir experiments/log/ablation1b_frozen_gs_dino \
  --ablation2a_dir experiments/log/ablation2a_mesh_gs_ssim \
  --ablation2b_dir experiments/log/ablation2b_mesh_gs_dino \
  --ablation2c_dir experiments/log/ablation2c_mesh_gs_combined \
  --output_dir experiments/log/comparison_all_ablations

# Metrics to compare:
# - loss_x, loss_v (geometry MSE)
# - loss_render (image reconstruction)
# - Chamfer distance (3D accuracy)
# - PSNR, SSIM (image quality)
# - Training time, GPU memory usage
```

### Expected Findings

**Ablation 0 vs 1:** Does render loss improve dynamics learning?
- Ablation 0 (no render loss) is baseline
- Ablation 1 (frozen GS + LBS) adds image-space supervision

**Ablation 1 vs 2:** Frozen vs trainable GS?
- Ablation 1 (frozen GS) has single optimizer, LBS deformation
- Ablation 2 (trainable GS) has dual optimizers, barycentric interpolation

**Ablation 2a vs 2b vs 2c:** Which loss works best?
- 2a (SSIM) should excel at structural alignment
- 2b (DINOv2) should be more robust to lighting/texture variation
- 2c (Combined) should perform best overall

---

## Additional Resources

### Key Files

| File | Purpose |
|------|---------|
| `mesh_gaussian_model.py` | Core mesh-constrained GS class |
| `build_cloth_mesh_bpa.py` | Ball Pivoting mesh construction |
| `build_cloth_mesh_poisson.py` | Poisson mesh reconstruction |
| `anchor_gaussians_to_mesh.py` | Gaussian-to-face anchoring |
| `render_loss_ablation2.py` | Dual-loss render loss module |
| `train_eval_ablation2.py` | Training script with dual backward |
| `config_ablation2_example.yaml` | Configuration template |
| `find_matching_episodes.py` | Episode + .splat matching |
| `check_gpu_capacity.py` | GPU capacity assessment |

### References

- **PGND Paper:** [Insert citation]
- **3D Gaussian Splatting:** Kerbl et al., SIGGRAPH 2023
- **PhysTwin (Render Loss):** [Insert citation]
- **DINOv2:** Oquab et al., ICLR 2024
- **Ball Pivoting Algorithm:** Bernardini et al., IEEE VIS 1999
- **Poisson Surface Reconstruction:** Kazhdan et al., SGP 2006

---

## FAQ

**Q: Can I use Ablation 2 without .splat files?**
A: No. Ablation 2 requires GS reconstructions for each training episode. Episodes without .splat files will be skipped.

**Q: How do I know if Ablation 2 is better than Ablation 1?**
A: Compare final metrics (Chamfer distance, PSNR, SSIM) on the same held-out test set. Lower Chamfer = better 3D accuracy, higher PSNR/SSIM = better image quality.

**Q: Can I fine-tune from Ablation 1 checkpoint?**
A: Partially. You can load the dynamics model checkpoint, but GS parameters must be initialized fresh (Ablation 1 has no GS gradients to resume from).

**Q: What if my mesh has holes?**
A: Ablation 2 uses BPA which may create holes in sparse regions. This is acceptable for now - the BPA vs Poisson comparison will be studied separately in Ablation 3. For Ablation 2, focus on loss function comparison (2a vs 2b vs 2c).

**Q: Should I train with point cloud dropout?**
A: Not for Ablation 2. All Ablation 2 experiments use full-density point clouds to isolate the effect of loss function choice. Point cloud dropout robustness will be studied separately in Ablation 4.

**Q: Which Ablation 2 variant should I run first?**
A: Start with Ablation 2a (SSIM loss) as it's the most direct comparison to Ablation 1. Then run 2b and 2c to understand the effect of different loss functions.

**Q: Can I use multiple cameras for rendering?**
A: Currently, render loss uses a single camera per iteration. The camera is selected via `render_camera_id` config. Multi-camera rendering (batching across views) is not yet implemented.

**Q: How do I visualize GS checkpoints?**
A: Use the evaluation script with `use_gs=True`:
```bash
python experiments/train/train_eval_ablation2.py \
  gpus=[0] \
  resume=True \
  train.resume_iteration=100000 \
  eval_only=True
```

---

## Contact

For issues or questions:
- Check `experiments/train/ABLATION2_TRAINING_GUIDE.md` (this file)
- Review W&B logs at https://wandb.ai/your-project
- Open an issue on GitHub: [your repo]

---

**Good luck with your training!** 🚀
