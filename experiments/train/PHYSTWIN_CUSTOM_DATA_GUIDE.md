# Using PhysTwin on Custom PGND Data for Future Prediction & MPC

## Overview

This guide shows how to:
1. Convert your PGND cloth data to PhysTwin format
2. Run PhysTwin optimization on your custom data
3. Use the optimized PhysTwin for future prediction
4. Integrate with MPC planning

## Why This Matters

**PhysTwin's Value Proposition:**
- Works on **ANY cloth** from just 10 seconds of observation
- No need for pre-trained model on similar cloths
- Adapts to new materials, textures, stiffness on-the-fly
- 25 min optimization → physics-based simulator ready for MPC

## Step 1: Convert PGND Data to PhysTwin Format

```bash
cd /home/fashionista/pgnd/experiments/train

# Convert one episode
python convert_pgnd_to_phystwin.py \
  --pgnd_episode /home/fashionista/pgnd/experiments/log/data_cloth/1224_cloth_fold_processed/episode_0000 \
  --output_dir /home/fashionista/PhysTwin/data/pgnd_cloth \
  --case_name pgnd_cloth_ep0

# This creates:
# /home/fashionista/PhysTwin/data/pgnd_cloth/pgnd_cloth_ep0/
# ├── color/          (RGB images from all cameras)
# ├── depth/          (Depth maps from all cameras)
# ├── mask/           (Segmentation masks)
# ├── calibrate.pkl   (Camera calibration)
# └── metadata.json   (Metadata)
```

## Step 2: Process Data (Generate Tracking & Point Clouds)

PhysTwin needs to extract:
- CoTracker3 tracking (pseudo-ground-truth particle trajectories)
- 3D point clouds from depth maps
- Shape priors from TRELLIS

```bash
cd /home/fashionista/PhysTwin

# Option A: If you want full processing (CoTracker3, shape priors)
python script_process_data.py  # Processes all cases in data/pgnd_cloth/

# Option B: Skip shape priors if you just want to test
# (You can use a simpler initialization)
```

**Note:** Since your PGND data already has ground-truth cloth meshes, you might be able to skip the shape prior generation and use your existing mesh as initialization!

## Step 3: Optimize Physics Parameters (Stage 1)

This is the key step - optimize spring-mass physics parameters to fit your cloth manipulation.

**Two approaches:**

### Approach A: Zero-Order CMA-ES Optimization (~12 min)
```bash
cd /home/fashionista/PhysTwin

python optimize_warp.py \
  --base_path ./data/pgnd_cloth \
  --case_name pgnd_cloth_ep0
```

This optimizes:
- Spring stiffness (Young's modulus)
- Damping coefficients
- Collision parameters
- Control forces

### Approach B: First-Order Gradient Optimization (~5 min, faster)
```bash
# If you want to skip CMA-ES and go straight to gradient-based:
python train_warp.py \
  --base_path ./data/pgnd_cloth \
  --case_name pgnd_cloth_ep0 \
  --train_frame 80 \
  --skip_cma

# Uses first 80 frames for optimization
```

**Output:**
```
/home/fashionista/PhysTwin/experiments_optimization/pgnd_cloth_ep0/
├── opt_params.pkl       # Optimized physics parameters
└── optimization.log     # Optimization progress
```

## Step 4: Train Appearance (Stage 2 - Optional for Physics Testing)

If you want realistic rendering (for MPC with visual feedback):

```bash
cd /home/fashionista/PhysTwin

python train_warp.py \
  --base_path ./data/pgnd_cloth \
  --case_name pgnd_cloth_ep0 \
  --train_frame 80
```

**Output:**
```
/home/fashionista/PhysTwin/experiments/pgnd_cloth_ep0/train/
├── best_99.pth           # Best physics + appearance model
└── sim_iter99.mp4        # Visualization video
```

## Step 5: Future Prediction

Now test if PhysTwin can predict future cloth states!

```bash
cd /home/fashionista/PhysTwin

python inference_warp.py \
  --base_path ./data/pgnd_cloth \
  --case_name pgnd_cloth_ep0 \
  --test_frame_start 80 \
  --test_frame_end 120
```

This will:
- Load optimized physics parameters
- Simulate from frame 80 onwards
- Compare predicted vs. ground-truth cloth state
- Output: `experiments/pgnd_cloth_ep0/inference.pkl`

**Evaluate prediction accuracy:**
```python
import pickle
import numpy as np

# Load inference results
with open('/home/fashionista/PhysTwin/experiments/pgnd_cloth_ep0/inference.pkl', 'rb') as f:
    results = pickle.load(f)

# Compute prediction error
pred_positions = results['predicted_positions']  # (T, N, 3)
gt_positions = results['gt_positions']           # (T, N, 3)

chamfer_dist = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=-1))
print(f"Future Prediction Chamfer Distance: {chamfer_dist:.6f}")
```

## Step 6: MPC Planning with PhysTwin

Now use the learned physics simulator for Model-Predictive Control!

### Create MPC Planning Script

```python
# /home/fashionista/pgnd/experiments/train/phystwin_mpc.py

import warp as wp
import torch
import pickle
import numpy as np
from pathlib import Path

class PhysTwinMPC:
    """
    Model-Predictive Control using PhysTwin's differentiable physics.
    """

    def __init__(self, phystwin_model_path, horizon=10):
        """
        Args:
            phystwin_model_path: Path to trained PhysTwin model
            horizon: MPC planning horizon (number of timesteps)
        """
        self.horizon = horizon

        # Load optimized physics parameters
        with open(phystwin_model_path, 'rb') as f:
            self.params = pickle.load(f)

        # Initialize Warp simulator
        wp.init()
        self.simulator = self.load_simulator()

    def load_simulator(self):
        """Load PhysTwin's spring-mass simulator with optimized params."""
        from qqtt.model.diff_simulator.spring_mass_warp import SpringMassWarp

        simulator = SpringMassWarp(
            spring_stiffness=self.params['spring_Y'],
            damping=self.params['damping'],
            # ... other physics parameters
        )
        return simulator

    def plan_action(self, current_state, goal_state, num_iters=50):
        """
        Plan control actions using MPC.

        Args:
            current_state: Current cloth particles (N, 3)
            goal_state: Target cloth configuration (N, 3)
            num_iters: Number of optimization iterations

        Returns:
            optimal_action: Best control action (gripper pose/velocity)
        """
        # Initialize action sequence
        actions = torch.randn(self.horizon, 6, requires_grad=True)  # (T, action_dim)

        optimizer = torch.optim.Adam([actions], lr=0.01)

        for iter in range(num_iters):
            optimizer.zero_grad()

            # Simulate forward with current action sequence
            state = current_state.clone()
            total_cost = 0.0

            for t in range(self.horizon):
                # Apply action to simulator
                state = self.simulator.step(state, actions[t])

                # Compute cost: distance to goal + action cost
                state_cost = torch.mean((state - goal_state) ** 2)
                action_cost = 0.01 * torch.mean(actions[t] ** 2)
                total_cost += state_cost + action_cost

            # Backpropagate through physics simulator
            total_cost.backward()
            optimizer.step()

            if iter % 10 == 0:
                print(f"MPC Iter {iter}: Cost = {total_cost.item():.6f}")

        # Return first action (MPC receding horizon)
        return actions[0].detach()

    def execute_mpc_loop(self, initial_state, goal_state, num_steps=50):
        """
        Execute full MPC control loop.
        """
        state = initial_state
        trajectory = [state.cpu().numpy()]

        for step in range(num_steps):
            # Plan optimal action
            action = self.plan_action(state, goal_state)

            # Execute action in simulator
            state = self.simulator.step(state, action)
            trajectory.append(state.cpu().numpy())

            # Check if goal reached
            dist_to_goal = torch.mean((state - goal_state) ** 2).item()
            print(f"Step {step}: Distance to goal = {dist_to_goal:.6f}")

            if dist_to_goal < 0.001:
                print("Goal reached!")
                break

        return np.array(trajectory)


# Example usage
if __name__ == '__main__':
    # Load PhysTwin model
    model_path = '/home/fashionista/PhysTwin/experiments/pgnd_cloth_ep0/train/best_99.pth'
    mpc = PhysTwinMPC(model_path, horizon=10)

    # Define initial and goal states
    initial_state = torch.randn(100, 3)  # 100 particles
    goal_state = initial_state + torch.randn(100, 3) * 0.1  # Perturbed goal

    # Run MPC
    trajectory = mpc.execute_mpc_loop(initial_state, goal_state)

    print(f"MPC completed: {len(trajectory)} steps")
```

## Step 7: Compare with Learning-Based (PGND)

Now compare PhysTwin's physics-based prediction with PGND's learned dynamics:

```bash
# Evaluate PGND on same episode
cd /home/fashionista/pgnd
python evaluate.py --episode episode_0000 --output pgnd_results.pkl

# Evaluate PhysTwin on same episode
cd /home/fashionista/PhysTwin
python inference_warp.py --case_name pgnd_cloth_ep0 --output phystwin_results.pkl

# Compare
python compare_predictions.py \
  --pgnd_results pgnd_results.pkl \
  --phystwin_results phystwin_results.pkl
```

## Key Metrics to Compare

| Metric | PGND (Learning) | PhysTwin (Physics) | Your Method |
|--------|-----------------|-------------------|-------------|
| **Optimization Time** | Training time (hours) | 25 min per cloth | Training time |
| **Future Prediction Error** | ? | ? | ? |
| **Long-Horizon Stability** | ? | ? | ? |
| **MPC Planning Speed** | Fast (neural net) | Slow (physics sim) | Fast |
| **Generalization** | Cross-cloth | Per-cloth only | Cross-cloth |

## Expected Insights

**What you'll learn:**

1. **PhysTwin's per-cloth accuracy**: How well does 25-min optimization work?
2. **Physics vs. Learning trade-off**: Is physics more stable for long horizons?
3. **MPC practicality**: Is PhysTwin fast enough for real-time control?
4. **Generalization gap**: How does per-cloth optimization compare to learned generalization?

## Troubleshooting

### Issue: Calibration conversion fails

**Solution**: Check your calibration format:
```python
import pickle
with open('/path/to/calibration/base.pkl', 'rb') as f:
    base = pickle.load(f)
print(base.keys())  # See what fields exist
```

### Issue: CoTracker3 processing is slow

**Solution**: Use your existing PGND particle trajectories as tracking:
```python
# Copy PGND trajectories to PhysTwin format
# (You already have ground-truth cloth mesh → can skip CoTracker)
```

### Issue: Physics optimization doesn't converge

**Solution**:
- Try different initialization (adjust spring stiffness range)
- Use more frames for optimization (--train_frame 120)
- Check if depth maps are clean

## Next Steps

Once PhysTwin is working on your data:

1. **Test on multiple cloth episodes**: Does same cloth need re-optimization?
2. **Compare future prediction**: PhysTwin vs. PGND vs. Your method
3. **MPC planning experiments**: Can PhysTwin enable better manipulation planning?
4. **Generalization benchmark**: Test if learning beats per-cloth optimization

The goal: Show that while PhysTwin is impressive per-cloth, learning-based (with rendering loss) wins on generalization + speed!
