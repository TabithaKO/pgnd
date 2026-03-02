"""
train_renderer.py — Standalone Neural Mesh Renderer Training (Ablation 2 v4d Phase 1)
======================================================================================

Trains the neural mesh renderer on high-quality baseline dynamics predictions.

Strategy:
    1. Load frozen baseline dynamics model
    2. Roll out training episodes, compute per-step MDE
    3. Filter to states where MDE < threshold (baseline got it right)
    4. Train neural mesh renderer on these filtered states
    5. Save converged renderer checkpoint for Phase 2

The key insight: train the renderer ONLY on states where the dynamics model
produces accurate geometry. This gives the renderer a clean training signal —
it learns "what does correct geometry look like when rendered?" without ever
seeing incorrect geometry. In Phase 2, the frozen renderer provides honest
gradient feedback to improve dynamics.

Usage:
    cd ~/pgnd/experiments/train
    python train_renderer.py \
        --baseline_ckpt cloth/train/ckpt/100000.pt \
        --episodes 162-241 \
        --mde_threshold 0.15 \
        --max_rollout_steps 15 \
        --num_iterations 50000 \
        --output_name cloth/renderer_v4d
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PGND path setup
# pgnd/ lives at ~/pgnd/pgnd/ but its __init__.py imports open3d which causes
# a numpy crash. We register a dummy pgnd module to skip __init__.py,
# then import submodules directly.
root = Path(__file__).resolve().parent

import types
_pgnd_root = root.parent.parent / 'pgnd'  # ~/pgnd/pgnd/
pgnd_dummy = types.ModuleType('pgnd')
pgnd_dummy.__path__ = [str(_pgnd_root)]
pgnd_dummy.__package__ = 'pgnd'
sys.modules['pgnd'] = pgnd_dummy

sys.path.append(str(root.parent.parent))  # ~/pgnd/ for pgnd.* submodules
sys.path.append(str(root.parent))         # ~/pgnd/experiments/ for train.*

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import get_root

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import warp as wp

from train_eval import transform_gripper_points, dataloader_wrapper

# Render pipeline imports
from train.render_loss import (
    GTImageLoader, setup_camera_for_render, masked_image_loss,
)
from train.render_loss_ablation2 import PGNDCoordinateTransform
from train.neural_mesh_renderer import (
    NeuralMeshRenderer, create_neural_mesh_renderer, project_vertex_colors,
)
from train.build_cloth_mesh import compute_mesh_from_particles

from diff_gaussian_rasterization import GaussianRasterizer


def parse_episode_range(s: str) -> List[int]:
    """Parse episode specification like '162-241' or '162,170,180'."""
    episodes = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return episodes


# =============================================================================
# Phase 1: Collect high-quality states from baseline rollouts
# =============================================================================

def rollout_and_filter(
    cfg,
    baseline_ckpt_path: str,
    episodes: List[int],
    mde_threshold: float,
    max_rollout_steps: int,
    torch_device: torch.device,
    wp_device: str,
) -> List[Dict]:
    """Roll out baseline model on episodes, filter states by MDE threshold.

    Returns:
        List of dicts, each containing:
            - 'episode': int
            - 'step': int
            - 'pred_particles': (N, 3) tensor in preprocessed space
            - 'gt_particles': (N, 3) tensor in preprocessed space
            - 'mde': float
    """
    log_root = get_root(__file__) / 'log'

    # Load baseline model (frozen)
    ckpt = torch.load(str(log_root / baseline_ckpt_path), map_location=torch_device)
    material = PGNDModel(cfg)
    material.to(torch_device)
    material.load_state_dict(ckpt['material'])
    material.requires_grad_(False)
    material.eval()

    if 'friction' in ckpt:
        friction = ckpt['friction']['mu'].reshape(-1, 1)
    else:
        friction = torch.tensor(cfg.model.friction.value, device=torch_device).reshape(-1, 1)

    source_dataset_root = log_root / str(cfg.train.source_dataset_name)

    all_states = []
    total_collected = 0
    total_evaluated = 0

    for episode in episodes:
        try:
            # Load dataset
            dataset = RealTeleopBatchDataset(
                cfg,
                dataset_root=log_root / cfg.train.dataset_name / 'state',
                source_data_root=source_dataset_root,
                device=torch_device,
                num_steps=cfg.sim.num_steps,
                eval_episode_name=f'episode_{episode:04d}',
            )
            dataloader = dataloader_wrapper(
                DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True),
                'dataset'
            )

            if cfg.sim.gripper_points:
                gripper_dataset = RealGripperDataset(cfg, device=torch_device)
                gripper_dataloader = dataloader_wrapper(
                    DataLoader(gripper_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True),
                    'gripper_dataset'
                )

            init_state, actions, gt_states, downsample_indices = next(dataloader)
            x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
            x = x.to(torch_device)
            v = v.to(torch_device)
            x_his = x_his.to(torch_device)
            v_his = v_his.to(torch_device)
            actions = actions.to(torch_device)

            if cfg.sim.gripper_points:
                gripper_points, _ = next(gripper_dataloader)
                gripper_points = gripper_points.to(torch_device)
                gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, actions)

            gt_x, gt_v = gt_states
            gt_x = gt_x.to(torch_device)

            batch_size = gt_x.shape[0]
            num_steps_total = min(gt_x.shape[1], max_rollout_steps)
            num_particles = gt_x.shape[2]

            if cfg.sim.gripper_points:
                num_gripper_particles = gripper_x.shape[2]
                num_particles_orig = num_particles
                num_particles = num_particles + num_gripper_particles

            cfg_copy = cfg.copy()
            cfg_copy.sim.num_steps = gt_x.shape[1]
            sim = CacheDiffSimWithFrictionBatch(cfg_copy, gt_x.shape[1], batch_size, wp_device, requires_grad=False)

            statics = StaticsBatch()
            statics.init(shape=(batch_size, num_particles), device=wp_device)
            statics.update_clip_bound(clip_bound)
            statics.update_enabled(enabled)
            colliders = CollidersBatch()

            if cfg.sim.gripper_points:
                num_grippers = 0
            else:
                num_grippers = cfg.sim.num_grippers
            colliders.init(shape=(batch_size, num_grippers), device=wp_device)
            if num_grippers > 0:
                colliders.initialize_grippers(actions[:, 0])

            enabled = enabled.to(torch_device)
            ep_collected = 0

            with torch.no_grad():
                for step in range(num_steps_total):
                    if num_grippers > 0:
                        colliders.update_grippers(actions[:, step])

                    if cfg.sim.gripper_points:
                        x = torch.cat([x, gripper_x[:, step]], dim=1)
                        v = torch.cat([v, gripper_v[:, step]], dim=1)
                        x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                        v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                        if enabled.shape[1] < num_particles:
                            enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                        statics.update_enabled(enabled.cpu())

                    pred = material(x, v, x_his, v_his, enabled)
                    x, v = sim(statics, colliders, step, x, v, friction, pred)

                    if cfg.sim.n_history > 0:
                        if cfg.sim.gripper_points:
                            x_his_particles = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None].detach()], dim=2)
                            v_his_particles = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None].detach()], dim=2)
                            x_his = x_his_particles.reshape(batch_size, num_particles_orig, -1)
                            v_his = v_his_particles.reshape(batch_size, num_particles_orig, -1)
                        else:
                            x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2)
                            v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2)
                            x_his = x_his.reshape(batch_size, num_particles, -1)
                            v_his = v_his.reshape(batch_size, num_particles, -1)

                    if cfg.sim.gripper_points:
                        pred_particles = x[:, :num_particles_orig]
                        x = pred_particles
                        v = v[:, :num_particles_orig]
                        enabled = enabled[:, :num_particles_orig]
                    else:
                        pred_particles = x

                    gt_particles = gt_x[0, step]
                    mde = torch.norm(pred_particles[0] - gt_particles, dim=-1).mean().item()
                    total_evaluated += 1

                    if mde < mde_threshold:
                        all_states.append({
                            'episode': episode,
                            'step': step,
                            'pred_particles': pred_particles[0].cpu(),
                            'gt_particles': gt_particles.cpu(),
                            'mde': mde,
                        })
                        ep_collected += 1
                        total_collected += 1

            print(f'  Episode {episode}: {ep_collected}/{num_steps_total} steps below MDE={mde_threshold:.3f}')

        except Exception as e:
            print(f'  Episode {episode}: FAILED — {e}')
            continue

    print(f'\nCollection complete: {total_collected}/{total_evaluated} states '
          f'({100*total_collected/max(total_evaluated,1):.1f}%) below MDE={mde_threshold}')

    return all_states


# =============================================================================
# Phase 1: Train neural mesh renderer
# =============================================================================

def train_renderer(
    cfg,
    filtered_states: List[Dict],
    num_iterations: int,
    output_dir: Path,
    torch_device: torch.device,
    lr: float = 1e-3,
    log_interval: int = 100,
    save_interval: int = 5000,
    debug_interval: int = 1000,
    resume_ckpt: str = None,
):
    """Train neural mesh renderer on filtered high-quality states.

    For each training iteration:
        1. Sample a random (episode, step) from filtered states
        2. Build mesh from predicted particles
        3. Load GT image, project vertex colors
        4. Forward through neural renderer MLP
        5. Rasterize, compute image loss vs GT
        6. Backward + step (renderer weights only)
    """
    log_root = get_root(__file__) / 'log'

    # Create renderer
    renderer = create_neural_mesh_renderer(
        hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
        use_vertex_colors=True, use_view_direction=True,
        device=str(torch_device),
    )
    renderer.train()

    # LPIPS loss
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(torch_device)
        lpips_fn.eval()
        use_lpips = True
        print('[train_renderer] LPIPS loss enabled')
    except ImportError:
        use_lpips = False
        print('[train_renderer] LPIPS not available, using L1+SSIM only')

    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-5)

    # Resume from checkpoint
    start_iteration = 1
    if resume_ckpt is not None:
        ckpt_path = log_root / resume_ckpt
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=torch_device)
            renderer.load_state_dict(ckpt['renderer'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_iteration = ckpt.get('iteration', 0) + 1
            # Advance scheduler to match
            for _ in range(start_iteration - 1):
                scheduler.step()
            print(f'[resume] Loaded checkpoint from {ckpt_path}, resuming at iteration {start_iteration}')
        else:
            print(f'[resume] WARNING: {ckpt_path} not found, starting fresh')

    # Group states by episode for efficient loading
    states_by_episode = defaultdict(list)
    for s in filtered_states:
        states_by_episode[s['episode']].append(s)

    # Pre-setup per-episode camera/GT loaders (cache to avoid re-loading)
    episode_cache = {}
    source_dataset_root = log_root / str(cfg.train.source_dataset_name)

    def get_episode_setup(episode: int):
        if episode in episode_cache:
            return episode_cache[episode]

        try:
            episode_data_path = source_dataset_root / f'episode_{episode:04d}'
            meta = np.loadtxt(str(episode_data_path / 'meta.txt'))

            with open(source_dataset_root / 'metadata.json') as f:
                metadata = json.load(f)
            entry = metadata[episode]
            source_data_dir = Path(entry['path'])
            recording_name = source_data_dir.parent.name
            source_episode_id = int(meta[0])

            n_history = int(cfg.sim.n_history)
            load_skip = int(cfg.train.dataset_load_skip_frame)
            ds_skip = int(cfg.train.dataset_skip_frame)
            source_frame_start = int(meta[1]) + n_history * load_skip * ds_skip

            episode_dir = log_root / 'data_cloth' / recording_name / f'episode_{source_episode_id:04d}'

            # Coordinate transform
            coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()

            # Camera
            calib_dir = episode_dir / 'calibration'
            intr = np.load(str(calib_dir / 'intrinsics.npy'))
            rvec = np.load(str(calib_dir / 'rvecs.npy'))
            tvec = np.load(str(calib_dir / 'tvecs.npy'))

            camera_id = 1
            R = cv2.Rodrigues(rvec[camera_id])[0]
            t = tvec[camera_id, :, 0]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            w2c = np.linalg.inv(c2w).astype(np.float32)

            cam_settings = {
                'w': 848, 'h': 480,
                'k': intr[camera_id],
                'w2c': w2c,
            }

            # GT loader
            gt_loader = GTImageLoader(
                episode_dir=episode_dir,
                source_frame_start=source_frame_start,
                camera_id=camera_id,
                image_size=(480, 848),
                skip_frame=load_skip * ds_skip,
            )

            setup = {
                'coord_transform': coord_transform,
                'cam_settings': cam_settings,
                'gt_loader': gt_loader,
            }
            episode_cache[episode] = setup
            return setup

        except Exception as e:
            print(f'  [setup] Episode {episode} failed: {e}')
            episode_cache[episode] = None
            return None

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(exist_ok=True)
    debug_dir = output_dir / 'debug'
    debug_dir.mkdir(exist_ok=True)

    print(f'\nTraining renderer for {num_iterations} iterations')
    print(f'  States: {len(filtered_states)} filtered samples')
    print(f'  Episodes: {len(states_by_episode)}')
    print(f'  Output: {output_dir}\n')

    # Initialize wandb
    try:
        import wandb
        wandb.init(
            project='pgnd-train',
            name=f'renderer_v4d_{time.strftime("%m%d_%H%M")}',
            config={
                'type': 'renderer_training_v4d',
                'num_iterations': num_iterations,
                'lr': lr,
                'num_states': len(filtered_states),
                'num_episodes': len(states_by_episode),
                'mde_threshold': filtered_states[0].get('mde', 0),  # just for reference
                'hidden_dim': 256,
                'n_hidden_layers': 4,
                'gaussians_per_face': 8,
                'use_lpips': use_lpips,
            },
        )
        use_wandb = True
        print('[wandb] Initialized')
    except Exception as e:
        print(f'[wandb] Not available: {e}')
        use_wandb = False

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_lpips = 0.0
    running_count = 0
    recent_renders = []  # buffer for multi-example wandb image logging
    t_start = time.time()

    for iteration in range(start_iteration, num_iterations + 1):
        # Sample random state
        idx = np.random.randint(len(filtered_states))
        state = filtered_states[idx]
        episode = state['episode']
        step = state['step']

        setup = get_episode_setup(episode)
        if setup is None:
            continue

        coord_transform = setup['coord_transform']
        cam_settings = setup['cam_settings']
        gt_loader = setup['gt_loader']

        # Load GT image at this step
        gt_image = gt_loader.load_frame(step)
        if gt_image is None:
            continue
        gt_mask = gt_loader.load_mask(step)

        particles = state['pred_particles'].to(torch_device)

        try:
            # Build mesh from particles
            mesh_data = compute_mesh_from_particles(particles, method='bpa')
            faces = mesh_data.face  # (3, N_faces)
            n_verts = mesh_data.pos.shape[0]
            vertices = particles[:n_verts]

            # Project vertex colors from GT image
            vertex_colors = project_vertex_colors(
                vertices_preproc=vertices.detach(),
                image=gt_image,
                cam_settings=cam_settings,
                coord_transform=coord_transform,
            )

            # Neural renderer forward
            rendered_image, debug_info = renderer(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                cam_settings=cam_settings,
                coord_transform=coord_transform,
            )

            # Compute loss components separately for logging
            if gt_mask is not None:
                pred_masked = rendered_image * gt_mask
                gt_masked = gt_image * gt_mask
            else:
                pred_masked = rendered_image
                gt_masked = gt_image

            loss_l1 = F.l1_loss(pred_masked, gt_masked)

            # SSIM via masked_image_loss with lambda_ssim=1.0 minus L1 component
            loss_combined = masked_image_loss(
                pred=rendered_image, gt=gt_image, mask=gt_mask, lambda_ssim=0.2,
            )
            # loss_combined = (1 - 0.2) * l1 + 0.2 * ssim_component
            # So ssim_component = (loss_combined - 0.8 * loss_l1) / 0.2
            loss_ssim = ((loss_combined - 0.8 * loss_l1) / 0.2).detach()

            loss = loss_combined  # Use the proper masked_image_loss

            # Add LPIPS
            loss_lpips_val = 0.0
            if use_lpips:
                lpips_val = lpips_fn(
                    rendered_image.unsqueeze(0) * 2 - 1,
                    gt_masked.unsqueeze(0) * 2 - 1,
                ).mean()
                loss_lpips_val = lpips_val.item()
                loss = 0.9 * loss + 0.1 * lpips_val

            # Backward + step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_l1 += loss_l1.item()
            running_ssim += loss_ssim.item()
            running_lpips += loss_lpips_val
            running_count += 1

            # Cache recent renders for multi-example logging
            if len(recent_renders) >= 4:
                recent_renders.pop(0)
            recent_renders.append({
                'rendered': rendered_image.detach().cpu().clamp(0, 1),
                'gt': gt_image.detach().cpu().clamp(0, 1),
                'mask': gt_mask.detach().cpu() if gt_mask is not None else None,
                'episode': episode,
                'step': step,
                'mde': state['mde'],
                'loss': loss.item(),
            })

        except Exception as e:
            if iteration <= 5:
                print(f'  [iter {iteration}] Error: {e}')
            continue

        # Logging
        if iteration % log_interval == 0:
            avg_loss = running_loss / max(running_count, 1)
            avg_l1 = running_l1 / max(running_count, 1)
            avg_ssim = running_ssim / max(running_count, 1)
            avg_lpips = running_lpips / max(running_count, 1)
            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(f'[renderer,iter {iteration:6d}/{num_iterations},'
                  f'loss={avg_loss:.6f},l1={avg_l1:.4f},ssim={avg_ssim:.4f},lpips={avg_lpips:.4f},'
                  f'lr={lr_now:.2e},'
                  f'ep={episode},step={step},mde={state["mde"]:.4f},'
                  f'time={elapsed:.0f}s]')

            if use_wandb:
                wandb.log({
                    'render_loss': avg_loss,
                    'loss_l1': avg_l1,
                    'loss_ssim': avg_ssim,
                    'loss_lpips': avg_lpips,
                    'lr': lr_now,
                    'sample_mde': state['mde'],
                    'iteration': iteration,
                }, step=iteration)

            running_loss = 0.0
            running_l1 = 0.0
            running_ssim = 0.0
            running_lpips = 0.0
            running_count = 0

        # Save checkpoint
        if iteration % save_interval == 0:
            ckpt_path = ckpt_dir / f'{iteration:06d}.pt'
            torch.save({
                'renderer': renderer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'num_states': len(filtered_states),
            }, str(ckpt_path))
            print(f'  Saved checkpoint: {ckpt_path}')

        # Debug visualization — log multiple examples
        if iteration % debug_interval == 0 and len(recent_renders) > 0:
            try:
                with torch.no_grad():
                    # Build a multi-row canvas: one row per recent render
                    examples = recent_renders[-4:]  # last 4 examples
                    h, w = examples[0]['rendered'].shape[1:]
                    n_examples = len(examples)

                    canvas = np.zeros((n_examples * (h + 25), w * 3 + 4, 3), dtype=np.float32)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    for row_idx, ex in enumerate(examples):
                        y_off = row_idx * (h + 25)
                        rendered_np = ex['rendered'].permute(1, 2, 0).numpy()
                        gt_np = ex['gt'].permute(1, 2, 0).numpy()
                        if ex['mask'] is not None:
                            gt_np = gt_np * ex['mask'].permute(1, 2, 0).numpy()

                        diff = np.abs(rendered_np - gt_np)
                        diff_amp = np.clip(diff * 5.0, 0, 1)

                        canvas[y_off+25:y_off+25+h, 0:w] = rendered_np
                        canvas[y_off+25:y_off+25+h, w+2:2*w+2] = gt_np
                        canvas[y_off+25:y_off+25+h, 2*w+4:3*w+4] = diff_amp

                        label = f'ep{ex["episode"]} s{ex["step"]} mde={ex["mde"]:.3f} loss={ex["loss"]:.4f}'
                        canvas_u8_row = (canvas[y_off:y_off+25] * 255).astype(np.uint8)
                        # Labels get written after full canvas conversion

                    canvas_u8 = (canvas * 255).astype(np.uint8)
                    for row_idx, ex in enumerate(examples):
                        y_off = row_idx * (h + 25)
                        label = f'ep{ex["episode"]} s{ex["step"]} mde={ex["mde"]:.3f} loss={ex["loss"]:.4f}'
                        cv2.putText(canvas_u8, f'Rendered | GT | Diff5x — {label}',
                                    (5, y_off + 18), font, 0.35, (255, 255, 255), 1)

                    debug_path = debug_dir / f'debug_{iteration:06d}.jpg'
                    cv2.imwrite(str(debug_path), cv2.cvtColor(canvas_u8, cv2.COLOR_RGB2BGR))

                    # Log to wandb
                    if use_wandb:
                        wandb.log({
                            'renders': wandb.Image(canvas_u8,
                                caption=f'iter={iteration}, {n_examples} examples'),
                        }, step=iteration)
            except Exception as e:
                if iteration <= 5000:
                    print(f'  [debug viz] Error: {e}')
                pass

    # Final save
    final_path = ckpt_dir / 'renderer_final.pt'
    torch.save({
        'renderer': renderer.state_dict(),
        'iteration': num_iterations,
        'num_states': len(filtered_states),
        'mde_threshold': args.mde_threshold if 'args' in dir() else 0.15,
    }, str(final_path))
    print(f'\nTraining complete. Final checkpoint: {final_path}')

    if use_wandb:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train neural mesh renderer on high-quality baseline states')
    parser.add_argument('--baseline_ckpt', type=str, required=True,
                        help='Path to baseline checkpoint (relative to log/)')
    parser.add_argument('--episodes', type=str, default='162-241',
                        help='Episode range (e.g. "162-241" or "162,170,180")')
    parser.add_argument('--mde_threshold', type=float, default=0.15,
                        help='MDE threshold for filtering states')
    parser.add_argument('--max_rollout_steps', type=int, default=15,
                        help='Max rollout steps per episode')
    parser.add_argument('--num_iterations', type=int, default=50000,
                        help='Number of renderer training iterations')
    parser.add_argument('--output_name', type=str, default='cloth/renderer_v4d',
                        help='Output directory name under log/')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to hydra config (default: baseline config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to renderer checkpoint to resume from (e.g. cloth/renderer_v4d/ckpt/050000.pt)')

    global args
    args = parser.parse_args()

    # Load config
    log_root = get_root(__file__) / 'log'
    if args.config:
        config_path = args.config
    else:
        # Use baseline's config
        baseline_dir = str(Path(args.baseline_ckpt).parent.parent)
        config_path = str(log_root / baseline_dir / 'hydra.yaml')

    print(f'Loading config from: {config_path}')
    cfg = OmegaConf.load(config_path)

    episodes = parse_episode_range(args.episodes)
    print(f'Episodes: {len(episodes)} ({episodes[0]}-{episodes[-1]})')
    print(f'MDE threshold: {args.mde_threshold}')
    print(f'Max rollout steps: {args.max_rollout_steps}')

    torch_device = torch.device('cuda')
    wp_device = 'cuda:0'

    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    wp.config.verify_autograd_array_access = True

    # Phase 1a: Collect filtered states
    print('\n' + '='*60)
    print('Phase 1a: Collecting high-quality baseline states')
    print('='*60)

    filtered_states = rollout_and_filter(
        cfg=cfg,
        baseline_ckpt_path=args.baseline_ckpt,
        episodes=episodes,
        mde_threshold=args.mde_threshold,
        max_rollout_steps=args.max_rollout_steps,
        torch_device=torch_device,
        wp_device=wp_device,
    )

    if len(filtered_states) < 100:
        print(f'\nERROR: Only {len(filtered_states)} states collected. '
              f'Try increasing --mde_threshold or --max_rollout_steps')
        return

    # Print distribution
    mde_vals = [s['mde'] for s in filtered_states]
    print(f'\nFiltered state MDE distribution:')
    print(f'  Mean:   {np.mean(mde_vals):.4f}')
    print(f'  Median: {np.median(mde_vals):.4f}')
    print(f'  Min:    {np.min(mde_vals):.4f}')
    print(f'  Max:    {np.max(mde_vals):.4f}')

    # Phase 1b: Train renderer
    print('\n' + '='*60)
    print('Phase 1b: Training neural mesh renderer')
    print('='*60)

    output_dir = log_root / args.output_name

    train_renderer(
        cfg=cfg,
        filtered_states=filtered_states,
        num_iterations=args.num_iterations,
        output_dir=output_dir,
        torch_device=torch_device,
        lr=args.lr,
        resume_ckpt=args.resume,
    )


if __name__ == '__main__':
    main()