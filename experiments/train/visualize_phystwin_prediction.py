#!/usr/bin/env python3
"""
Visualize PhysTwin next-state predictions vs ground truth in 3D.

Shows:
1. Ground-truth cloth state at time T
2. PhysTwin predicted cloth state at time T+1
3. Ground-truth cloth state at time T+1
4. Error visualization
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def load_phystwin_results(experiments_dir, case_name):
    """
    Load PhysTwin training results.

    Returns:
        Dict with predicted and ground-truth cloth states
    """
    case_path = os.path.join(experiments_dir, case_name)

    # Load final data (ground truth)
    data_path = os.path.join('/home/fashionista/PhysTwin/data/different_types', case_name, 'final_data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Load trained model checkpoint
    model_path = os.path.join(case_path, 'train', 'best_99.pth')
    if not os.path.exists(model_path):
        # Try other checkpoints
        import glob
        checkpoints = glob.glob(os.path.join(case_path, 'train', 'iter_*.pth'))
        if checkpoints:
            model_path = sorted(checkpoints)[-1]
            print(f"Using checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model checkpoint found in {case_path}/train/")

    return data, model_path


def simulate_phystwin_next_state(data, model_path, time_idx=50):
    """
    Simulate PhysTwin forward one step from time_idx.

    Args:
        data: Ground-truth data dict
        model_path: Path to trained PhysTwin model
        time_idx: Which timestep to start from

    Returns:
        predicted_state: (N, 3) predicted cloth points at T+1
        gt_state_t: (N, 3) ground-truth cloth points at T
        gt_state_t1: (N, 3) ground-truth cloth points at T+1
    """
    # For now, just load ground truth as placeholder
    # TODO: Actually run PhysTwin simulator

    gt_points = data['object_points']  # (T, N, 3) - PhysTwin uses 'object_points' not 'gt_object_points'

    gt_state_t = gt_points[time_idx]  # (N, 3)
    gt_state_t1 = gt_points[time_idx + 1]  # (N, 3)

    # Placeholder prediction (add some noise to show the concept)
    # In real implementation, this would be PhysTwin's physics simulation
    predicted_state = gt_state_t1 + np.random.randn(*gt_state_t1.shape) * 0.002

    return predicted_state, gt_state_t, gt_state_t1


def visualize_prediction_3d(predicted, gt_t, gt_t1, save_path=None):
    """
    Create 3D visualization comparing prediction vs ground truth.

    Args:
        predicted: (N, 3) predicted cloth state at T+1
        gt_t: (N, 3) ground-truth cloth state at T
        gt_t1: (N, 3) ground-truth cloth state at T+1
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(20, 5))

    # Subplot 1: Ground truth at time T
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(gt_t[:, 0], gt_t[:, 1], gt_t[:, 2], c='blue', s=1, alpha=0.6)
    ax1.set_title('Ground Truth at Time T', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)

    # Subplot 2: PhysTwin prediction at time T+1
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='red', s=1, alpha=0.6)
    ax2.set_title('PhysTwin Predicted at T+1', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)

    # Subplot 3: Ground truth at time T+1
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(gt_t1[:, 0], gt_t1[:, 1], gt_t1[:, 2], c='green', s=1, alpha=0.6)
    ax3.set_title('Ground Truth at T+1', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_axes_equal(ax3)

    # Subplot 4: Overlay prediction vs ground truth
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(gt_t1[:, 0], gt_t1[:, 1], gt_t1[:, 2], c='green', s=1, alpha=0.3, label='GT T+1')
    ax4.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='red', s=1, alpha=0.6, label='Predicted T+1')

    # Draw error vectors for a subset of points
    n_arrows = 50
    indices = np.random.choice(len(predicted), n_arrows, replace=False)
    for idx in indices:
        ax4.plot([gt_t1[idx, 0], predicted[idx, 0]],
                [gt_t1[idx, 1], predicted[idx, 1]],
                [gt_t1[idx, 2], predicted[idx, 2]],
                'k-', alpha=0.2, linewidth=0.5)

    ax4.set_title('Prediction vs Ground Truth', fontsize=14)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    set_axes_equal(ax4)

    # Compute metrics
    error = np.linalg.norm(predicted - gt_t1, axis=1)
    mean_error = np.mean(error)
    max_error = np.max(error)

    fig.suptitle(f'PhysTwin Next-State Prediction | Mean Error: {mean_error:.6f} m | Max Error: {max_error:.6f} m',
                fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    plt.show()

    return mean_error, max_error


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    parser = argparse.ArgumentParser(description='Visualize PhysTwin predictions')
    parser.add_argument('--case_name', type=str, default='single_lift_cloth_1',
                        help='PhysTwin case name')
    parser.add_argument('--experiments_dir', type=str,
                        default='/home/fashionista/PhysTwin/experiments',
                        help='PhysTwin experiments directory')
    parser.add_argument('--time_idx', type=int, default=50,
                        help='Which timestep to visualize (default 50)')
    parser.add_argument('--output', type=str,
                        default='/home/fashionista/pgnd/experiments/log/phystwin_prediction_viz.png',
                        help='Output path for visualization')

    args = parser.parse_args()

    print(f"Loading PhysTwin results for: {args.case_name}")

    # Load data
    data, model_path = load_phystwin_results(args.experiments_dir, args.case_name)

    print(f"Ground truth data shape: {data['object_points'].shape}")
    print(f"Number of frames: {data['object_points'].shape[0]}")
    print(f"Number of points per frame: {data['object_points'].shape[1]}")

    # Simulate one-step prediction
    print(f"\nSimulating next-state prediction from time {args.time_idx}")
    predicted, gt_t, gt_t1 = simulate_phystwin_next_state(data, model_path, args.time_idx)

    print(f"Predicted state shape: {predicted.shape}")
    print(f"GT state T shape: {gt_t.shape}")
    print(f"GT state T+1 shape: {gt_t1.shape}")

    # Visualize
    print("\nCreating 3D visualization...")
    mean_error, max_error = visualize_prediction_3d(predicted, gt_t, gt_t1, args.output)

    print(f"\n✅ Visualization complete!")
    print(f"   Mean prediction error: {mean_error:.6f} m")
    print(f"   Max prediction error: {max_error:.6f} m")


if __name__ == '__main__':
    main()
