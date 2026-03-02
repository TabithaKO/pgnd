# Running PhysTwin Training

## Quick Start

```bash
# 1. Navigate to PhysTwin directory
cd /home/fashionista/PhysTwin

# 2. Activate conda environment
conda activate pgnd

# 3. Login to wandb (if not already logged in)
wandb login

# 4. Run training
python train_warp.py \
  --base_path ./data/different_types \
  --case_name single_lift_cloth_1 \
  --train_frame 80
```

## Training Details

**Expected Duration**: ~25 minutes (100 iterations × 15 seconds)
**GPU Usage**: ~2-4 GB
**Wandb Project**: `final_pipeline`
**Wandb Run Name**: `single_lift_cloth_1`

## Monitor Progress

### In the terminal:
Watch for log messages like:
```
[Train]: Iteration: 0, Loss: 0.00012334067609221097
[Train]: Iteration: 1, Loss: 0.00013588351441308366
```

### On Wandb:
1. Go to: https://wandb.ai/tabby-research/final_pipeline
2. Look for run: `single_lift_cloth_1`
3. Metrics logged:
   - `loss`: Total loss (chamfer + tracking)
   - `chamfer_loss`: 3D geometry loss
   - `track_loss`: Tracking error
   - `video`: Visualization videos

## Run in Background (tmux recommended)

```bash
# Option 1: Using tmux (recommended)
tmux new -s phystwin
cd /home/fashionista/PhysTwin
conda activate pgnd
python train_warp.py \
  --base_path ./data/different_types \
  --case_name single_lift_cloth_1 \
  --train_frame 80

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t phystwin

# Option 2: Using nohup
cd /home/fashionista/PhysTwin
conda activate pgnd
nohup python train_warp.py \
  --base_path ./data/different_types \
  --case_name single_lift_cloth_1 \
  --train_frame 80 > phystwin.log 2>&1 &

# Monitor progress:
tail -f phystwin.log
```

## Train on Different Cloth Cases

Available cases in `data/different_types/`:
- `single_lift_cloth_1` (default)
- `single_lift_cloth_3`
- `single_lift_cloth_4`
- `single_clift_cloth_1`
- `single_clift_cloth_3`
- `double_lift_cloth_1`
- `double_lift_cloth_3`

```bash
# Example: Train on different case
python train_warp.py \
  --base_path ./data/different_types \
  --case_name double_lift_cloth_1 \
  --train_frame 80
```

## Check GPU Usage

```bash
# Monitor GPU while training
watch -n 1 nvidia-smi

# Or check current usage
nvidia-smi
```

## Troubleshooting

### Issue: Wandb not logging

**Solution**: Make sure you're logged in to wandb:
```bash
wandb login
# Enter your API key when prompted
```

### Issue: CUDA out of memory

**Solution**: PhysTwin uses ~2-4 GB. If you're running Ablation 1 in parallel:
```bash
# Check current GPU usage
nvidia-smi

# If needed, stop other training first
pkill -f train_eval  # Stop PGND training
```

### Issue: Module not found errors

**Solution**: Make sure you're using the `pgnd` environment (not `phystwin`):
```bash
conda activate pgnd
```

## Output Files

Training creates:
```
/home/fashionista/PhysTwin/
├── experiments/single_lift_cloth_1/
│   ├── train/
│   │   ├── best_99.pth              # Best model checkpoint
│   │   ├── iter_*.pth               # Periodic checkpoints
│   │   └── sim_iter*.mp4            # Visualization videos
│   ├── inv_phy_log.log              # Training logs
│   └── inference.pkl                # Final predictions
└── wandb/
    └── run-*/                        # Wandb logs
```

## Applied Fixes (Already Done)

The following compatibility fixes have been applied to the PhysTwin code:

1. **pytorch3d removed** (not needed for training)
   - File: `gs_render.py`

2. **Warp clamp API fixed** (keyword args → positional)
   - File: `qqtt/model/diff_simulator/spring_mass_warp.py`
   - Lines: 124, 278-279, 329-330

3. **Torch→Warp conversion with dtypes**
   - File: `qqtt/model/diff_simulator/spring_mass_warp.py`
   - Function: `set_controller_target()`

These fixes are persistent - you don't need to reapply them.

## Training Multiple Cases

To train on all cloth cases:
```bash
cd /home/fashionista/PhysTwin
conda activate pgnd

# Train all cases sequentially (~3 hours total)
python script_train.py
```

This will train on all 8 cloth cases and log to wandb.

## Comparison with PGND

| Aspect | PGND (Ablation 1) | PhysTwin |
|--------|------------------|----------|
| Training Time | ~30 hours | ~25 minutes |
| GPU Memory | ~12 GB | ~2-4 GB |
| Iterations | 100K | 100 |
| Supervision | Ground-truth trajectories | Pseudo-tracking |
| Can Run Together | ✅ Yes (16 GB GPU) | ✅ Yes |

## Next Steps

After training completes:
1. Check wandb for loss curves and videos
2. Run evaluation: `python evaluate_chamfer.py --case_name single_lift_cloth_1`
3. Compare with PGND baseline metrics
