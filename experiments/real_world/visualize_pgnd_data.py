#!/usr/bin/env python3
"""
PGND Dataset Visualizer
========================
Visualize components of the PGND training data:
  - Point clouds (particles) with velocity arrows
  - Robot end-effector trajectories
  - Gripper states over time
  - Raw recording data (RGB, depth, masks) if available
  - Gaussian splatting reconstructions if available

Usage:
    # Visualize a sub-episode's point cloud trajectory
    python visualize_pgnd_data.py \
        --data-dir ~/pgnd/experiments/log/data/cloth_merged/sub_episodes_v \
        --episode 0 \
        --mode pointcloud

    # Visualize robot trajectory
    python visualize_pgnd_data.py ... --mode robot

    # Visualize everything for an episode
    python visualize_pgnd_data.py ... --mode all

    # Export frames as images (no GUI needed)
    python visualize_pgnd_data.py ... --mode pointcloud --export ~/viz_output

    # Browse raw processed data (before sub-episode extraction)
    python visualize_pgnd_data.py \
        --raw-dir ~/pgnd/experiments/log/data_cloth/0112_cloth_processed/episode_0000 \
        --mode raw
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Data loading
# =============================================================================

def load_sub_episode(data_dir: Path, episode_id: int):
    """Load a sub-episode from sub_episodes_v/ format."""
    ep_dir = data_dir / f"episode_{episode_id:04d}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode not found: {ep_dir}")

    data = {}

    # Particle trajectories
    traj = np.load(str(ep_dir / "traj.npz"))
    data["xyz"] = traj["xyz"]  # (T, N, 3)
    data["vel"] = traj["v"]    # (T, N, 3)

    # Camera indices
    cam_path = ep_dir / "cam_indices.txt"
    if cam_path.exists():
        data["cam_indices"] = np.loadtxt(str(cam_path))  # (N,)

    # Robot data
    eef_traj_path = ep_dir / "eef_traj.txt"
    if eef_traj_path.exists():
        data["eef_traj"] = np.loadtxt(str(eef_traj_path))  # (T, n_arms*3)

    eef_rot_path = ep_dir / "eef_rot.txt"
    if eef_rot_path.exists():
        data["eef_rot"] = np.loadtxt(str(eef_rot_path))  # (T, n_arms*9)

    eef_grip_path = ep_dir / "eef_gripper.txt"
    if eef_grip_path.exists():
        data["eef_gripper"] = np.loadtxt(str(eef_grip_path))  # (T, n_arms)

    # Meta
    meta_path = ep_dir / "meta.txt"
    if meta_path.exists():
        data["meta"] = np.loadtxt(str(meta_path))

    return data


def load_raw_episode(raw_dir: Path):
    """Load raw processed episode data (pcd_clean/, robot/, etc.)."""
    data = {}

    # Point clouds
    pcd_dir = raw_dir / "pcd_clean"
    if pcd_dir.exists():
        pcd_files = sorted(pcd_dir.glob("*.npz"))
        data["pcd_files"] = pcd_files
        if pcd_files:
            sample = np.load(str(pcd_files[0]))
            data["pcd_keys"] = list(sample.keys())

    # RGB images
    for cam_id in range(4):
        rgb_dir = raw_dir / f"camera_{cam_id}" / "rgb"
        if rgb_dir.exists():
            rgb_files = sorted(rgb_dir.glob("*.jpg"))
            if rgb_files:
                data[f"rgb_cam{cam_id}"] = rgb_files

    # Depth
    for cam_id in range(4):
        depth_dir = raw_dir / f"camera_{cam_id}" / "depth"
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob("*.png"))
            if depth_files:
                data[f"depth_cam{cam_id}"] = depth_files

    # Masks
    for cam_id in range(4):
        mask_dir = raw_dir / f"camera_{cam_id}" / "mask"
        if mask_dir.exists():
            mask_files = sorted(mask_dir.glob("*.png"))
            if mask_files:
                data[f"mask_cam{cam_id}"] = mask_files

    # Robot
    robot_dir = raw_dir / "robot"
    if robot_dir.exists():
        data["robot_files"] = sorted(robot_dir.glob("*.txt"))

    # Gaussian splatting
    gs_dir = raw_dir / "gs"
    if gs_dir.exists():
        data["gs_files"] = sorted(gs_dir.glob("*.splat"))

    # Calibration
    calib_dir = raw_dir / "calibration"
    if calib_dir.exists():
        data["has_calibration"] = True
        if (calib_dir / "intrinsics.npy").exists():
            data["intrinsics"] = np.load(str(calib_dir / "intrinsics.npy"))
        if (calib_dir / "rvecs.npy").exists():
            data["rvecs"] = np.load(str(calib_dir / "rvecs.npy"))
        if (calib_dir / "tvecs.npy").exists():
            data["tvecs"] = np.load(str(calib_dir / "tvecs.npy"))

    return data


# =============================================================================
# Point cloud visualization
# =============================================================================

def plot_pointcloud_frame(ax, xyz, vel=None, cam_indices=None,
                          eef_pos=None, title="", vel_scale=0.05):
    """Plot a single frame's point cloud on a 3D axis."""
    ax.cla()

    if cam_indices is not None:
        unique_cams = np.unique(cam_indices)
        colors_map = plt.cm.Set1(np.linspace(0, 1, max(int(unique_cams.max()) + 1, 2)))
        colors = colors_map[cam_indices.astype(int)]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='steelblue', s=1, alpha=0.6)

    # Velocity arrows (subsample for visibility)
    if vel is not None and len(xyz) > 0:
        vel_norm = np.linalg.norm(vel, axis=1)
        if vel_norm.max() > 1e-6:
            n_arrows = min(200, len(xyz))
            idx = np.random.choice(len(xyz), n_arrows, replace=False)
            ax.quiver(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
                      vel[idx, 0], vel[idx, 1], vel[idx, 2],
                      length=vel_scale, color='red', alpha=0.5, linewidth=0.5)

    # Robot end-effector positions
    if eef_pos is not None:
        n_arms = len(eef_pos) // 3
        arm_colors = ['green', 'orange', 'purple', 'cyan']
        for arm in range(n_arms):
            pos = eef_pos[arm * 3:(arm + 1) * 3]
            ax.scatter(*pos, c=arm_colors[arm % len(arm_colors)],
                       s=100, marker='x', linewidths=3,
                       label=f'Arm {arm}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Consistent axis limits
    if len(xyz) > 0:
        center = xyz.mean(axis=0)
        max_range = max(xyz.max(axis=0) - xyz.min(axis=0)) / 2 * 1.2
        max_range = max(max_range, 0.1)
        for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(center[i] - max_range, center[i] + max_range)


def visualize_pointcloud_sequence(data: dict, output_dir: Path,
                                  frame_step: int = 5, max_frames: int = 50):
    """Generate point cloud visualization frames."""
    xyz = data["xyz"]  # (T, N, 3)
    vel = data.get("vel")
    cam_indices = data.get("cam_indices")
    eef_traj = data.get("eef_traj")

    T, N, _ = xyz.shape
    frames_to_plot = list(range(0, T, frame_step))[:max_frames]

    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Rendering {len(frames_to_plot)} point cloud frames...")

    for i, t in enumerate(frames_to_plot):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        eef_pos = eef_traj[t] if eef_traj is not None else None
        v = vel[t] if vel is not None else None

        plot_pointcloud_frame(
            ax, xyz[t], v, cam_indices, eef_pos,
            title=f"Frame {t}/{T} | {N} particles")

        if eef_pos is not None:
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(pc_dir / f"frame_{t:04d}.png", dpi=100)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(frames_to_plot)} frames rendered")

    print(f"  Saved to {pc_dir}")
    return pc_dir


# =============================================================================
# Robot trajectory visualization
# =============================================================================

def visualize_robot_trajectory(data: dict, output_dir: Path):
    """Visualize robot end-effector trajectory and gripper states."""
    robot_dir = output_dir / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)

    eef_traj = data.get("eef_traj")
    eef_gripper = data.get("eef_gripper")

    if eef_traj is None:
        print("  No robot trajectory data found")
        return

    T = eef_traj.shape[0]
    n_arms = eef_traj.shape[1] // 3
    time_axis = np.arange(T)

    # 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    arm_colors = ['green', 'orange', 'purple', 'cyan']
    for arm in range(n_arms):
        pos = eef_traj[:, arm * 3:(arm + 1) * 3]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                c=arm_colors[arm % len(arm_colors)],
                linewidth=2, label=f'Arm {arm}')
        ax.scatter(*pos[0], c=arm_colors[arm % len(arm_colors)],
                   s=100, marker='o', edgecolors='black')  # start
        ax.scatter(*pos[-1], c=arm_colors[arm % len(arm_colors)],
                   s=100, marker='s', edgecolors='black')  # end

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'End-Effector Trajectories ({T} timesteps)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(robot_dir / "eef_trajectory_3d.png", dpi=150)
    plt.close(fig)

    # Per-axis over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axis_labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes, axis_labels)):
        for arm in range(n_arms):
            pos = eef_traj[:, arm * 3 + i]
            ax.plot(time_axis, pos, c=arm_colors[arm % len(arm_colors)],
                    label=f'Arm {arm}')
        ax.set_ylabel(f'{label} (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Timestep')
    axes[0].set_title('End-Effector Position Over Time')
    plt.tight_layout()
    plt.savefig(robot_dir / "eef_trajectory_axes.png", dpi=150)
    plt.close(fig)

    # Gripper states
    if eef_gripper is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        for arm in range(eef_gripper.shape[1] if eef_gripper.ndim > 1 else 1):
            grip = eef_gripper[:, arm] if eef_gripper.ndim > 1 else eef_gripper
            ax.plot(time_axis, grip, c=arm_colors[arm % len(arm_colors)],
                    linewidth=2, label=f'Arm {arm}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Gripper State (raw)')
        ax.set_title('Gripper States Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(robot_dir / "gripper_states.png", dpi=150)
        plt.close(fig)

    # Velocity magnitude
    if "vel" in data:
        vel = data["vel"]  # (T, N, 3)
        vel_norms = np.linalg.norm(vel, axis=2)  # (T, N)
        mean_vel = vel_norms.mean(axis=1)
        max_vel = vel_norms.max(axis=1)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_axis, mean_vel, label='Mean velocity', color='steelblue')
        ax.fill_between(time_axis, 0, max_vel, alpha=0.2, color='steelblue',
                         label='Max velocity')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Particle Velocity Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(robot_dir / "particle_velocity.png", dpi=150)
        plt.close(fig)

    print(f"  Robot visualization saved to {robot_dir}")


# =============================================================================
# Summary statistics
# =============================================================================

def print_episode_summary(data: dict, episode_id: int):
    """Print summary statistics for an episode."""
    xyz = data["xyz"]
    T, N, _ = xyz.shape

    print(f"\n{'='*60}")
    print(f"Episode {episode_id} Summary")
    print(f"{'='*60}")
    print(f"  Timesteps: {T}")
    print(f"  Particles: {N}")

    # Spatial extent
    all_pts = xyz.reshape(-1, 3)
    print(f"  Spatial range:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"    {axis}: [{all_pts[:, i].min():.4f}, {all_pts[:, i].max():.4f}]"
              f" (span: {all_pts[:, i].max() - all_pts[:, i].min():.4f}m)")

    # Velocity stats
    if "vel" in data:
        vel = data["vel"]
        vel_norms = np.linalg.norm(vel, axis=2)
        print(f"  Velocity:")
        print(f"    Mean: {vel_norms.mean():.4f} m/s")
        print(f"    Max:  {vel_norms.max():.4f} m/s")
        print(f"    Std:  {vel_norms.std():.4f} m/s")

    # Camera distribution
    if "cam_indices" in data:
        cam_idx = data["cam_indices"]
        unique, counts = np.unique(cam_idx, return_counts=True)
        print(f"  Camera distribution:")
        for c, n in zip(unique, counts):
            print(f"    Camera {int(c)}: {n} particles ({100*n/N:.1f}%)")

    # Robot data
    if "eef_traj" in data:
        eef = data["eef_traj"]
        n_arms = eef.shape[1] // 3
        print(f"  Robot: {n_arms} arm(s)")
        for arm in range(n_arms):
            pos = eef[:, arm*3:(arm+1)*3]
            travel = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
            print(f"    Arm {arm}: travel distance = {travel:.4f}m")

    if "eef_gripper" in data:
        grip = data["eef_gripper"]
        print(f"  Gripper range: [{grip.min():.1f}, {grip.max():.1f}]")

    if "meta" in data:
        meta = data["meta"]
        print(f"  Meta: recording={int(meta[0])}, "
              f"frames={int(meta[1])}-{int(meta[2])}")

    print()


# =============================================================================
# Raw data visualization
# =============================================================================

def visualize_raw_data(raw_dir: Path, output_dir: Path,
                       frame_id: int = 0, max_cams: int = 4):
    """Visualize raw processed episode data (RGB, depth, masks, point clouds)."""
    import cv2

    data = load_raw_episode(raw_dir)
    raw_out = output_dir / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    print(f"\nRaw episode data in {raw_dir}:")
    for key, val in data.items():
        if isinstance(val, list):
            print(f"  {key}: {len(val)} files")
        elif isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")

    # RGB + Mask + Depth composite per camera
    n_cams = 0
    for cam_id in range(max_cams):
        if f"rgb_cam{cam_id}" in data:
            n_cams += 1

    if n_cams > 0:
        fig, axes = plt.subplots(n_cams, 3, figsize=(18, 5 * n_cams))
        if n_cams == 1:
            axes = axes[None, :]

        cam_count = 0
        for cam_id in range(max_cams):
            rgb_key = f"rgb_cam{cam_id}"
            if rgb_key not in data:
                continue

            rgb_files = data[rgb_key]
            fid = min(frame_id, len(rgb_files) - 1)

            # RGB
            rgb = cv2.imread(str(rgb_files[fid]))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            axes[cam_count, 0].imshow(rgb)
            axes[cam_count, 0].set_title(f'Camera {cam_id} - RGB (frame {fid})')
            axes[cam_count, 0].axis('off')

            # Mask
            mask_key = f"mask_cam{cam_id}"
            if mask_key in data and fid < len(data[mask_key]):
                mask = cv2.imread(str(data[mask_key][fid]), cv2.IMREAD_UNCHANGED)
                axes[cam_count, 1].imshow(mask, cmap='gray')
                axes[cam_count, 1].set_title(f'Camera {cam_id} - Mask')
            else:
                axes[cam_count, 1].text(0.5, 0.5, 'No mask', ha='center',
                                         va='center', transform=axes[cam_count, 1].transAxes)
            axes[cam_count, 1].axis('off')

            # Depth
            depth_key = f"depth_cam{cam_id}"
            if depth_key in data and fid < len(data[depth_key]):
                depth = cv2.imread(str(data[depth_key][fid]), cv2.IMREAD_UNCHANGED)
                axes[cam_count, 2].imshow(depth, cmap='turbo')
                axes[cam_count, 2].set_title(f'Camera {cam_id} - Depth')
            else:
                axes[cam_count, 2].text(0.5, 0.5, 'No depth', ha='center',
                                         va='center', transform=axes[cam_count, 2].transAxes)
            axes[cam_count, 2].axis('off')

            cam_count += 1

        plt.tight_layout()
        plt.savefig(raw_out / f"cameras_frame_{fid:06d}.png", dpi=150)
        plt.close(fig)
        print(f"  Camera composite saved")

    # Point cloud from pcd_clean
    if "pcd_files" in data and len(data["pcd_files"]) > 0:
        fid = min(frame_id, len(data["pcd_files"]) - 1)
        pcd = np.load(str(data["pcd_files"][fid]))

        print(f"  PCD keys: {list(pcd.keys())}")

        fig = plt.figure(figsize=(14, 10))

        # Point cloud colored by position
        ax1 = fig.add_subplot(121, projection='3d')
        pts = pcd["pts"]
        if "colors" in pcd:
            colors = pcd["colors"]
            if colors.max() > 1:
                colors = colors / 255.0
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1)
        else:
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='steelblue', s=1)
        ax1.set_title(f'Point Cloud (frame {fid}, {len(pts)} pts)')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

        # Point cloud colored by velocity magnitude
        ax2 = fig.add_subplot(122, projection='3d')
        if "vels" in pcd:
            vels = pcd["vels"]
            vel_mag = np.linalg.norm(vels, axis=1)
            sc = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                             c=vel_mag, cmap='hot', s=1, vmin=0,
                             vmax=min(vel_mag.max(), 0.5))
            plt.colorbar(sc, ax=ax2, shrink=0.6, label='Velocity (m/s)')
            ax2.set_title(f'Velocity Magnitude')
        else:
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='gray', s=1)
            ax2.set_title('No velocity data')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

        plt.tight_layout()
        plt.savefig(raw_out / f"pointcloud_frame_{fid:06d}.png", dpi=150)
        plt.close(fig)
        print(f"  Point cloud visualization saved")

    print(f"  Raw visualizations saved to {raw_out}")


# =============================================================================
# Dataset overview
# =============================================================================

def visualize_dataset_overview(data_dir: Path, output_dir: Path):
    """Generate overview statistics for the entire dataset."""
    overview_dir = output_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(data_dir.glob("episode_*"))
    n_episodes = len(episodes)
    print(f"\nDataset overview: {n_episodes} episodes in {data_dir}")

    n_particles_list = []
    n_timesteps_list = []
    travel_distances = []
    vel_means = []

    for ep_dir in episodes:
        traj_path = ep_dir / "traj.npz"
        if not traj_path.exists():
            continue
        traj = np.load(str(traj_path))
        xyz = traj["xyz"]
        T, N, _ = xyz.shape
        n_particles_list.append(N)
        n_timesteps_list.append(T)

        vel = traj["v"]
        vel_means.append(np.linalg.norm(vel, axis=2).mean())

        eef_path = ep_dir / "eef_traj.txt"
        if eef_path.exists():
            eef = np.loadtxt(str(eef_path))
            if eef.ndim > 1:
                pos = eef[:, :3]
                travel = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
                travel_distances.append(travel)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(n_particles_list, bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of particles')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Particles per Episode (N={n_episodes})')

    axes[0, 1].hist(n_timesteps_list, bins=30, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Number of timesteps')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Timesteps per Episode')

    axes[1, 0].hist(vel_means, bins=30, color='seagreen', edgecolor='black')
    axes[1, 0].set_xlabel('Mean velocity (m/s)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Mean Particle Velocity per Episode')

    if travel_distances:
        axes[1, 1].hist(travel_distances, bins=30, color='orchid', edgecolor='black')
        axes[1, 1].set_xlabel('EEF travel distance (m)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('End-Effector Travel Distance per Episode')
    else:
        axes[1, 1].text(0.5, 0.5, 'No robot data', ha='center', va='center',
                         transform=axes[1, 1].transAxes)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Dataset: {data_dir.name} ({n_episodes} episodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(overview_dir / "dataset_overview.png", dpi=150)
    plt.close(fig)

    # Summary stats
    print(f"  Episodes: {n_episodes}")
    print(f"  Particles: {np.mean(n_particles_list):.0f} ± {np.std(n_particles_list):.0f} "
          f"(range: {np.min(n_particles_list)}-{np.max(n_particles_list)})")
    print(f"  Timesteps: {np.mean(n_timesteps_list):.0f} ± {np.std(n_timesteps_list):.0f}")
    print(f"  Mean velocity: {np.mean(vel_means):.4f} ± {np.std(vel_means):.4f} m/s")
    if travel_distances:
        print(f"  EEF travel: {np.mean(travel_distances):.4f} ± {np.std(travel_distances):.4f} m")

    print(f"  Overview saved to {overview_dir}")


# =============================================================================
# Video generation
# =============================================================================

def generate_video(raw_dir: Path, output_dir: Path, cam_id: int = 0,
                   fps: int = 1, frame_step: int = 1):
    """Generate a video with 2x2 grid: RGB, Mask, Depth, Point Cloud at given fps."""
    import cv2
    import subprocess

    data = load_raw_episode(raw_dir)
    video_dir = output_dir / "video_frames"
    video_dir.mkdir(parents=True, exist_ok=True)

    rgb_key = f"rgb_cam{cam_id}"
    mask_key = f"mask_cam{cam_id}"
    depth_key = f"depth_cam{cam_id}"

    if rgb_key not in data:
        print(f"ERROR: No RGB data for camera {cam_id}")
        return

    rgb_files = data[rgb_key]
    n_frames = len(rgb_files)
    has_mask = mask_key in data
    has_depth = depth_key in data
    has_pcd = "pcd_files" in data and len(data.get("pcd_files", [])) > 0

    # Determine which frames to render
    frames_to_render = list(range(0, n_frames, frame_step))
    print(f"\nGenerating video for camera {cam_id}: {len(frames_to_render)} frames "
          f"(of {n_frames} total, step={frame_step})")

    # Precompute point cloud axis limits if we have pcd data
    pcd_bounds = None
    if has_pcd:
        pcd_files = data["pcd_files"]
        # Sample a few to get stable bounds
        sample_indices = np.linspace(0, len(pcd_files) - 1, min(10, len(pcd_files))).astype(int)
        all_pts = []
        for si in sample_indices:
            pcd = np.load(str(pcd_files[si]))
            all_pts.append(pcd["pts"])
        all_pts = np.concatenate(all_pts, axis=0)
        center = all_pts.mean(axis=0)
        max_range = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2
        max_range = max(max_range, 0.1)
        pcd_bounds = (center, max_range)

    for i, fid in enumerate(frames_to_render):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # --- Top-left: RGB ---
        rgb = cv2.imread(str(rgb_files[fid]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f'RGB - cam{cam_id} frame {fid}', fontsize=11)
        axes[0, 0].axis('off')

        # --- Top-right: Mask ---
        if has_mask and fid < len(data[mask_key]):
            mask = cv2.imread(str(data[mask_key][fid]), cv2.IMREAD_UNCHANGED)
            axes[0, 1].imshow(mask, cmap='gray')
            axes[0, 1].set_title('Mask', fontsize=11)
        else:
            axes[0, 1].text(0.5, 0.5, 'No mask', ha='center', va='center',
                            transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].axis('off')

        # --- Bottom-left: Depth ---
        if has_depth and fid < len(data[depth_key]):
            depth = cv2.imread(str(data[depth_key][fid]), cv2.IMREAD_UNCHANGED)
            axes[1, 0].imshow(depth, cmap='turbo')
            axes[1, 0].set_title('Depth', fontsize=11)
        else:
            axes[1, 0].text(0.5, 0.5, 'No depth', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].axis('off')

        # --- Bottom-right: Point Cloud ---
        if has_pcd:
            pcd_files = data["pcd_files"]
            pcd_fid = min(fid, len(pcd_files) - 1)
            pcd = np.load(str(pcd_files[pcd_fid]))
            pts = pcd["pts"]

            # Remove the 2D axis and replace with 3D
            axes[1, 1].remove()
            ax3d = fig.add_subplot(2, 2, 4, projection='3d')

            if "colors" in pcd:
                colors = pcd["colors"]
                if colors.max() > 1:
                    colors = colors / 255.0
                ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.5, alpha=0.7)
            elif "vels" in pcd:
                vel_mag = np.linalg.norm(pcd["vels"], axis=1)
                ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vel_mag,
                             cmap='hot', s=0.5, vmin=0, vmax=0.3)
            else:
                ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='steelblue', s=0.5)

            ax3d.set_title(f'Point Cloud ({len(pts)} pts)', fontsize=11)
            ax3d.set_xlabel('X', fontsize=8)
            ax3d.set_ylabel('Y', fontsize=8)
            ax3d.set_zlabel('Z', fontsize=8)
            ax3d.tick_params(labelsize=6)

            if pcd_bounds:
                center, max_range = pcd_bounds
                ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
                ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
                ax3d.set_zlim(center[2] - max_range, center[2] + max_range)
        else:
            axes[1, 1].text(0.5, 0.5, 'No point cloud', ha='center', va='center',
                            transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(video_dir / f"frame_{i:06d}.png", dpi=120)
        plt.close(fig)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{len(frames_to_render)} frames rendered")

    print(f"  All {len(frames_to_render)} frames rendered to {video_dir}")

    # Stitch with ffmpeg
    video_path = output_dir / f"cam{cam_id}_overview.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(video_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(video_path)
    ]
    print(f"  Encoding video at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Video saved: {video_path}")
    else:
        print(f"  ffmpeg error: {result.stderr[-500:]}")
        print(f"  Frames still available in {video_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="PGND Dataset Visualizer")

    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to sub_episodes_v/ directory")
    p.add_argument("--raw-dir", type=str, default=None,
                   help="Path to raw processed episode (with pcd_clean/, camera_*/)")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode ID to visualize")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame ID for raw visualization")
    p.add_argument("--mode", type=str, default="all",
                   choices=["pointcloud", "robot", "raw", "overview", "all", "video"],
                   help="Visualization mode")
    p.add_argument("--cam", type=int, default=0,
                   help="Camera ID for video mode")
    p.add_argument("--fps", type=int, default=1,
                   help="FPS for video output")
    p.add_argument("--export", type=str, default="./pgnd_viz",
                   help="Output directory for visualizations")
    p.add_argument("--frame-step", type=int, default=5,
                   help="Step between point cloud frames")
    p.add_argument("--max-frames", type=int, default=50,
                   help="Max number of point cloud frames to render")

    args = p.parse_args()

    output_dir = Path(args.export).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "raw":
        if args.raw_dir is None:
            print("ERROR: --raw-dir required for raw mode")
            return
        raw_dir = Path(args.raw_dir).expanduser()
        visualize_raw_data(raw_dir, output_dir, frame_id=args.frame)
        return

    if args.mode == "video":
        if args.raw_dir is None:
            print("ERROR: --raw-dir required for video mode")
            return
        raw_dir = Path(args.raw_dir).expanduser()
        generate_video(raw_dir, output_dir, cam_id=args.cam, fps=args.fps,
                       frame_step=args.frame_step)
        return

    if args.data_dir is None:
        print("ERROR: --data-dir required")
        return

    data_dir = Path(args.data_dir).expanduser()

    if args.mode == "overview":
        visualize_dataset_overview(data_dir, output_dir)
        return

    # Load episode
    print(f"Loading episode {args.episode} from {data_dir}...")
    data = load_sub_episode(data_dir, args.episode)
    print_episode_summary(data, args.episode)

    ep_out = output_dir / f"episode_{args.episode:04d}"

    if args.mode in ["pointcloud", "all"]:
        print("Generating point cloud visualizations...")
        visualize_pointcloud_sequence(data, ep_out,
                                      frame_step=args.frame_step,
                                      max_frames=args.max_frames)

    if args.mode in ["robot", "all"]:
        print("Generating robot trajectory visualizations...")
        visualize_robot_trajectory(data, ep_out)

    print(f"\nAll visualizations saved to {ep_out}")


if __name__ == "__main__":
    main()