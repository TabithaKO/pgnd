# PhysTwin Baseline Experiments

This directory contains PhysTwin baseline results for comparison with PGND's learned dynamics approach.

## PhysTwin Learning Paradigm

**PhysTwin learns physics by matching observations:**

1. **Input**: Real cloth trajectories from PGND data (T=0 to T=N)
2. **Optimization**: Fits physics parameters to match observed behavior
   - Spring stiffness (Y-modulus)
   - Damping coefficients (drag, dashpot)
   - Collision properties (elasticity, friction)
   - Particle connectivity (radius, max neighbors)
3. **Prediction**: Uses learned parameters to simulate future states via differentiable physics

**Key Difference from PGND:**
- **PhysTwin**: Learns interpretable physics parameters → simulates forward
- **PGND**: Learns direct state→next_state mapping via neural rendering

## Coordinate System Issue - RESOLVED ✅

### Problem
Initial RGB overlay visualizations showed misaligned coordinates because:
- Used predictions from `sub_episodes_v/episode_0112` (point clouds only)
- Overlaid on RGB from `0114_cloth6_processed/episode_0000` (different episode)
- **Different coordinate systems** → points appeared in wrong locations

### Solution
Use data from the **same** episode for both point clouds and RGB:
1. Generate point clouds from RGB-D data in same episode
2. Apply correct coordinate transformations:
   ```
   Depth → Camera coords (K matrix)
         → World coords (E_c2w = inverse of E_w2c)
         → Camera coords (R, t for projection)
         → Image pixels (K matrix)
   ```
3. Validate projection with [coordinate_system_demo](coordinate_system_demo/)

## Directory Structure

```
phystwin_results/
├── README.md (this file)
├── coordinate_system_demo/     # Validation of 3D→2D projection
│   ├── projection_demo_frame_0000.png
│   ├── projection_demo_frame_0500.png
│   ├── projection_demo_frame_1000.png
│   ├── projection_demo_frame_1500.png
│   └── projection_demo_frame_2000.png
└── (future: training results, predictions, comparisons)
```

## Coordinate System Demo Results

✅ **5 visualizations** showing 5,000 3D points each correctly projecting onto RGB images

**Validation:**
- All 5000 points valid and in camera view
- Points align perfectly with cloth regions in RGB
- Confirms coordinate transformations work correctly

**Files:**
- Frame 0, 500, 1000, 1500, 2000 from episode 0114_cloth6_processed/episode_0000
- Each shows: Original RGB (masked) + 3D points projected to 2D (green overlay)

## Next Steps

1. **Train PhysTwin on PGND data**
   - Convert PGND processed episodes to PhysTwin format
   - Run physics optimization to learn cloth parameters
   - Expected: ~25 min per episode for training

2. **Generate Predictions**
   - Use learned physics to predict future cloth states
   - Run on test set (held-out frames)

3. **RGB Overlay Visualization**
   - Project predicted 3D points onto actual RGB images
   - Compare with ground truth overlays
   - Use validated coordinate transformations from this demo

4. **Baseline Comparison**
   - PhysTwin (physics-based) vs PGND (learned dynamics)
   - Metrics: Chamfer distance, per-point errors
   - Analyze where physics priors help vs. learned approach

## Key Files

- `demo_correct_projection.py` - Validates coordinate transformations
- `run_phystwin_processed_episode.sh` - Full pipeline for single episode
- `phystwin_stress_test.sh` - Single episode comprehensive test
- `phystwin_maximum_stress_test.sh` - 20 episode stress test
- `visualize_rgb_overlay.py` - Overlay predictions on RGB images

## Notes

- PhysTwin requires per-scene optimization (no cross-scene generalization)
- Using pretrained cloth parameters from `single_lift_cloth_3` as initialization
- PGND data: 4504 frames, 8000 points/frame, 70/30 train/test split
- All experiments use camera 0 for visualization
