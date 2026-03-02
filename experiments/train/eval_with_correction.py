"""
eval_with_correction.py — Inference-Time Visual Correction (Cloth-Splatting Style)
==================================================================================

Implements predict-update filtering at test time:
    PREDICT:  particles_t+1 = dynamics(particles_t, action_t)
    RENDER:   image_pred = frozen_renderer(particles_t+1)
    OBSERVE:  image_gt = actual camera frame at t+1
    CORRECT:  delta = -alpha * dL_image/d_particles  (gradient descent)
    USE:      particles_t+1_corrected for next dynamics step

Complementary to Phase 2 training-time render loss:
- Phase 2 improves the dynamics MODEL weights (permanent)
- This corrects individual PREDICTIONS at test time (per-step, temporary)

Usage:
    cd ~/pgnd/experiments/train
    python eval_with_correction.py \
        --dynamics_ckpt cloth/train/ckpt/100000.pt \
        --renderer_ckpt cloth/renderer_v4d/ckpt/200000.pt \
        --episodes 610-650 \
        --correction_steps 5 \
        --correction_lr 0.001 \
        --max_correction 0.02 \
        --output_name cloth/eval_corrected

    # With Phase 2 dynamics:
    python eval_with_correction.py \
        --dynamics_ckpt cloth/phase2_v4d/ckpt/040000.pt \
        --renderer_ckpt cloth/renderer_v4d/ckpt/200000.pt \
        --episodes 610-650 \
        --correction_steps 10 \
        --correction_lr 0.002 \
        --output_name cloth/eval_phase2_corrected

    # No-correction baseline (for fair comparison):
    python eval_with_correction.py \
        --dynamics_ckpt cloth/train/ckpt/100000.pt \
        --renderer_ckpt cloth/renderer_v4d/ckpt/200000.pt \
        --episodes 610-650 \
        --correction_steps 0 \
        --output_name cloth/eval_no_correction
"""

import argparse
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
from tqdm import trange

# ── PGND path setup (same as train_renderer.py) ──
root = Path(__file__).resolve().parent

import types
_pgnd_root = root.parent.parent / 'pgnd'
pgnd_dummy = types.ModuleType('pgnd')
pgnd_dummy.__path__ = [str(_pgnd_root)]
pgnd_dummy.__package__ = 'pgnd'
sys.modules['pgnd'] = pgnd_dummy

sys.path.append(str(root.parent.parent))
sys.path.append(str(root.parent))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import get_root

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import warp as wp

from train_eval import transform_gripper_points, dataloader_wrapper

# Render pipeline
from train.render_loss import GTImageLoader, setup_camera_for_render, masked_image_loss
from train.render_loss_ablation2 import PGNDCoordinateTransform
from train.neural_mesh_renderer import (
    NeuralMeshRenderer, create_neural_mesh_renderer, project_vertex_colors,
)
from train.build_cloth_mesh import compute_mesh_from_particles

from diff_gaussian_rasterization import GaussianRasterizer


def parse_episode_range(s: str) -> List[int]:
    episodes = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            episodes.extend(range(int(a), int(b) + 1))
        else:
            episodes.append(int(part))
    return episodes


# =====================================================================
# Core: visual correction via gradient descent on particle positions
# =====================================================================

def visual_correction(
    particles: torch.Tensor,            # (N, 3) predicted positions, preproc space
    renderer: NeuralMeshRenderer,       # frozen neural mesh renderer
    gt_image: torch.Tensor,             # (3, H, W) GT RGB
    gt_mask: Optional[torch.Tensor],    # (1, H, W) cloth mask or None
    cam_settings: dict,                 # camera intrinsics/extrinsics
    coord_transform: PGNDCoordinateTransform,
    correction_steps: int = 5,
    correction_lr: float = 0.001,
    max_correction: float = 0.02,       # max displacement per particle (preproc space)
    lambda_ssim: float = 0.2,
    reg_weight: float = 0.1,            # isometric regularization weight
) -> Tuple[torch.Tensor, dict]:
    """
    Refine predicted particle positions using photometric loss.
    Follows Cloth-Splatting Algorithm 1 (predict-update).

    Key design: mesh topology is built ONCE from initial particles
    (not differentiable). Only vertex positions carry gradients through
    the frozen differentiable renderer.

    Returns:
        corrected_particles: (N, 3) refined positions
        info: dict with correction statistics
    """
    N = particles.shape[0]
    device = particles.device

    # ── Build mesh topology ONCE (not differentiable) ──
    try:
        mesh_data = compute_mesh_from_particles(particles.detach(), method='bpa')
    except Exception as e:
        return particles, {'status': 'mesh_failed', 'error': str(e)}

    faces = mesh_data.face           # (2, N_edges) for edge list, or (3, N_faces)
    n_verts = mesh_data.pos.shape[0]

    # The correction offset delta — this is the ONLY optimized variable
    delta = torch.zeros(n_verts, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=correction_lr)

    initial_loss = None
    final_loss = None

    for step in range(correction_steps):
        optimizer.zero_grad()

        # Apply correction to vertices (mesh was built from first n_verts particles)
        corrected_verts = particles[:n_verts].detach() + delta

        # Project vertex colors from GT image
        # (detached positions for projection — only render path carries grad)
        vertex_colors = project_vertex_colors(
            vertices_preproc=corrected_verts.detach(),
            image=gt_image,
            cam_settings=cam_settings,
            coord_transform=coord_transform,
        )

        # Render through neural mesh renderer
        try:
            rendered_image, _ = renderer(
                vertices=corrected_verts,
                faces=faces,
                vertex_colors=vertex_colors,
                cam_settings=cam_settings,
                coord_transform=coord_transform,
            )
        except Exception as e:
            if step == 0:
                return particles, {'status': 'render_failed', 'error': str(e)}
            break

        # Photometric loss
        loss_image = masked_image_loss(
            pred=rendered_image, gt=gt_image, mask=gt_mask,
            lambda_ssim=lambda_ssim,
        )

        # Isometric regularization: penalize large corrections
        loss_reg = (delta ** 2).sum(dim=-1).mean()
        loss = loss_image + reg_weight * loss_reg

        if initial_loss is None:
            initial_loss = loss_image.item()

        loss.backward()
        optimizer.step()

        # Clamp correction magnitude
        with torch.no_grad():
            norms = delta.norm(dim=-1)  # (n_verts,)
            exceed_mask = norms > max_correction
            if exceed_mask.any():
                scale = max_correction / norms[exceed_mask]
                delta.data[exceed_mask] *= scale.unsqueeze(-1)

        final_loss = loss_image.item()

    # Apply correction to full particle set
    corrected_particles = particles.detach().clone()
    corrected_particles[:n_verts] = particles[:n_verts].detach() + delta.detach()

    info = {
        'status': 'ok',
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': (initial_loss - final_loss) / max(initial_loss, 1e-8) if initial_loss else 0,
        'mean_correction': delta.detach().norm(dim=-1).mean().item(),
        'max_correction_actual': delta.detach().norm(dim=-1).max().item(),
        'n_verts_corrected': n_verts,
        'n_particles_total': N,
    }
    return corrected_particles, info


# =====================================================================
# Per-episode rendering setup
# =====================================================================

def setup_episode_rendering(
    cfg, episode: int, log_root: Path, camera_id: int = 1,
) -> Optional[dict]:
    """Resolve episode metadata and create camera/GT loader."""
    source_dataset_root = log_root / str(cfg.train.source_dataset_name)
    episode_data_path = source_dataset_root / f'episode_{episode:04d}'

    try:
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

        coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()

        calib_dir = episode_dir / 'calibration'
        intr = np.load(str(calib_dir / 'intrinsics.npy'))
        rvec = np.load(str(calib_dir / 'rvecs.npy'))
        tvec = np.load(str(calib_dir / 'tvecs.npy'))

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

        gt_loader = GTImageLoader(
            episode_dir=episode_dir,
            source_frame_start=source_frame_start,
            camera_id=camera_id,
            image_size=(480, 848),
            skip_frame=load_skip * ds_skip,
        )

        return {
            'coord_transform': coord_transform,
            'cam_settings': cam_settings,
            'gt_loader': gt_loader,
            'episode_dir': episode_dir,
        }

    except Exception as e:
        print(f'  [setup] Episode {episode} rendering setup failed: {e}')
        import traceback; traceback.print_exc()
        return None


# =====================================================================
# Eval episode with correction loop
# =====================================================================

def eval_episode_with_correction(
    cfg,
    material: nn.Module,
    friction: torch.Tensor,
    renderer: NeuralMeshRenderer,
    episode: int,
    torch_device: torch.device,
    wp_device,
    log_root: Path,
    output_dir: Path,
    correction_steps: int = 5,
    correction_lr: float = 0.001,
    max_correction: float = 0.02,
    camera_id: int = 1,
    save_debug: bool = True,
) -> Optional[np.ndarray]:
    """Eval one episode with visual correction.

    Returns (n_steps, 2) array: columns are [MDE_uncorrected, MDE_corrected].
    """
    source_dataset_root = log_root / str(cfg.train.source_dataset_name)

    render_setup = setup_episode_rendering(cfg, episode, log_root, camera_id)
    use_correction = (render_setup is not None) and (correction_steps > 0)

    # ── Load eval data ──
    try:
        eval_dataset = RealTeleopBatchDataset(
            cfg,
            dataset_root=log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root,
            device=torch_device,
            num_steps=cfg.sim.num_steps,
            eval_episode_name=f'episode_{episode:04d}',
        )
        eval_dataloader = dataloader_wrapper(
            DataLoader(eval_dataset, batch_size=1, shuffle=False,
                       num_workers=0, pin_memory=True),
            'dataset'
        )
        if cfg.sim.gripper_points:
            gripper_dataset = RealGripperDataset(cfg, device=torch_device)
            gripper_dataloader = dataloader_wrapper(
                DataLoader(gripper_dataset, batch_size=1, shuffle=False,
                           num_workers=0, pin_memory=True),
                'gripper_dataset'
            )

        init_state, actions, gt_states, downsample_indices = next(eval_dataloader)
    except Exception as e:
        print(f'  Episode {episode}: dataset load failed — {e}')
        return None

    x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
    x = x.to(torch_device)
    v = v.to(torch_device)
    x_his = x_his.to(torch_device)
    v_his = v_his.to(torch_device)
    actions = actions.to(torch_device)

    if cfg.sim.gripper_points:
        gripper_points, _ = next(gripper_dataloader)
        gripper_points = gripper_points.to(torch_device)
        gripper_x, gripper_v, gripper_mask = transform_gripper_points(
            cfg, gripper_points, actions)

    gt_x, gt_v = gt_states
    gt_x = gt_x.to(torch_device)
    gt_v = gt_v.to(torch_device)

    batch_size = gt_x.shape[0]
    num_steps_total = gt_x.shape[1]
    num_particles = gt_x.shape[2]
    assert batch_size == 1

    if cfg.sim.gripper_points:
        num_gripper_particles = gripper_x.shape[2]
        num_particles_orig = num_particles
        num_particles = num_particles + num_gripper_particles

    sim = CacheDiffSimWithFrictionBatch(
        cfg, num_steps_total, batch_size, wp_device, requires_grad=True)

    statics = StaticsBatch()
    statics.init(shape=(batch_size, num_particles), device=wp_device)
    statics.update_clip_bound(clip_bound)
    statics.update_enabled(enabled)
    colliders = CollidersBatch()

    material.eval()

    num_grippers = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
    colliders.init(shape=(batch_size, num_grippers), device=wp_device)
    if num_grippers > 0:
        colliders.initialize_grippers(actions[:, 0])

    enabled = enabled.to(torch_device)
    enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

    metrics_uncorrected = []
    metrics_corrected = []

    if save_debug:
        debug_dir = output_dir / 'debug' / f'episode_{episode:04d}'
        debug_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════ ROLLOUT LOOP ═══════════════
    for step in trange(num_steps_total, desc=f'ep {episode}', leave=False):
        if num_grippers > 0:
            colliders.update_grippers(actions[:, step])

        if cfg.sim.gripper_points:
            x = torch.cat([x, gripper_x[:, step]], dim=1)
            v = torch.cat([v, gripper_v[:, step]], dim=1)
            pad_x = torch.zeros(
                gripper_x.shape[0], gripper_x.shape[2],
                cfg.sim.n_history * 3, device=x_his.device, dtype=x_his.dtype)
            pad_v = torch.zeros_like(pad_x)
            x_his = torch.cat([x_his, pad_x], dim=1)
            v_his = torch.cat([v_his, pad_v], dim=1)
            if enabled.shape[1] < num_particles:
                enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
            statics.update_enabled(enabled.cpu())

        # ── PREDICT ──
        with torch.no_grad():
            pred = material(x, v, x_his, v_his, enabled)
            x_pred, v_pred = sim(
                statics, colliders, step, x, v, friction, pred)

        if cfg.sim.gripper_points:
            x_cloth = x_pred[:, :num_particles_orig]
            v_cloth = v_pred[:, :num_particles_orig]
        else:
            x_cloth = x_pred
            v_cloth = v_pred

        # MDE without correction
        gt_particles = gt_x[0, step]
        mde_uncorr = torch.norm(
            x_cloth[0] - gt_particles, dim=-1).mean().item()
        metrics_uncorrected.append(mde_uncorr)

        # ── CORRECT ──
        x_corrected = x_cloth.clone()
        step_info = {'status': 'skipped'}

        if use_correction:
            gt_image = render_setup['gt_loader'].load_frame(step)
            gt_mask_img = render_setup['gt_loader'].load_mask(step)

            if gt_image is not None:
                x_single, step_info = visual_correction(
                    particles=x_cloth[0],
                    renderer=renderer,
                    gt_image=gt_image,
                    gt_mask=gt_mask_img,
                    cam_settings=render_setup['cam_settings'],
                    coord_transform=render_setup['coord_transform'],
                    correction_steps=correction_steps,
                    correction_lr=correction_lr,
                    max_correction=max_correction,
                )
                x_corrected = x_single.unsqueeze(0)

        mde_corr = torch.norm(
            x_corrected[0] - gt_particles, dim=-1).mean().item()
        metrics_corrected.append(mde_corr)

        # ── USE corrected state for next step ──
        x = x_corrected
        v = v_cloth

        # Update history
        if cfg.sim.n_history > 0:
            if cfg.sim.gripper_points:
                xh = x_his[:, :num_particles_orig].reshape(
                    batch_size, num_particles_orig, -1, 3)
                vh = v_his[:, :num_particles_orig].reshape(
                    batch_size, num_particles_orig, -1, 3)
                x_his = torch.cat(
                    [xh[:, :, 1:], x[:, :, None].detach()], dim=2
                ).reshape(batch_size, num_particles_orig, -1)
                v_his = torch.cat(
                    [vh[:, :, 1:], v[:, :, None].detach()], dim=2
                ).reshape(batch_size, num_particles_orig, -1)
            else:
                xh = x_his.reshape(batch_size, num_particles, -1, 3)
                vh = v_his.reshape(batch_size, num_particles, -1, 3)
                x_his = torch.cat(
                    [xh[:, :, 1:], x[:, :, None].detach()], dim=2
                ).reshape(batch_size, num_particles, -1)
                v_his = torch.cat(
                    [vh[:, :, 1:], v[:, :, None].detach()], dim=2
                ).reshape(batch_size, num_particles, -1)

        if cfg.sim.gripper_points:
            enabled = enabled[:, :num_particles_orig]

        # Debug
        if save_debug and step % 5 == 0 and step_info.get('status') == 'ok':
            _save_debug(debug_dir / f'step_{step:03d}.txt',
                        step, mde_uncorr, mde_corr, step_info)

    # ═══════════════ EPISODE SUMMARY ═══════════════
    metrics_u = np.array(metrics_uncorrected)
    metrics_c = np.array(metrics_corrected)

    print(f'  Episode {episode}:')
    for s in [5, 10, 15, 20, 30]:
        if s < len(metrics_u):
            d = (metrics_u[s] - metrics_c[s]) / max(metrics_u[s], 1e-8) * 100
            print(f'    Step {s:2d}: uncorr={metrics_u[s]:.4f}  '
                  f'corr={metrics_c[s]:.4f}  delta={d:+.1f}%')

    metric_dir = output_dir / 'metric' / f'episode_{episode:04d}'
    metric_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(metric_dir / 'mde_uncorrected.txt'), metrics_u)
    np.savetxt(str(metric_dir / 'mde_corrected.txt'), metrics_c)

    return np.stack([metrics_u, metrics_c], axis=-1)


def _save_debug(path, step, mde_u, mde_c, info):
    with open(str(path), 'w') as f:
        d = (mde_u - mde_c) / max(mde_u, 1e-8) * 100
        f.write(f'step={step}\n')
        f.write(f'mde_uncorrected={mde_u:.6f}\n')
        f.write(f'mde_corrected={mde_c:.6f}\n')
        f.write(f'improvement={d:+.1f}%\n')
        for k, val in info.items():
            f.write(f'{k}={val}\n')


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Eval with inference-time visual correction')
    parser.add_argument('--dynamics_ckpt', type=str, required=True,
                        help='Dynamics checkpoint relative to log/')
    parser.add_argument('--renderer_ckpt', type=str, required=True,
                        help='Frozen renderer checkpoint relative to log/')
    parser.add_argument('--episodes', type=str, default='610-650')
    parser.add_argument('--correction_steps', type=int, default=5,
                        help='Gradient steps per correction (0=none)')
    parser.add_argument('--correction_lr', type=float, default=0.001)
    parser.add_argument('--max_correction', type=float, default=0.02)
    parser.add_argument('--output_name', type=str,
                        default='cloth/eval_corrected')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--camera_id', type=int, default=1)
    parser.add_argument('--no_debug', action='store_true')

    args = parser.parse_args()

    log_root = get_root(__file__) / 'log'
    torch_device = torch.device('cuda')
    wp_device = 'cuda:0'

    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    # ── Config ──
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        dynamics_dir = str(Path(args.dynamics_ckpt).parent.parent)
        cfg = OmegaConf.load(str(log_root / dynamics_dir / 'hydra.yaml'))

    # ── Dynamics model ──
    print(f'Loading dynamics: {args.dynamics_ckpt}')
    ckpt = torch.load(
        str(log_root / args.dynamics_ckpt), map_location=torch_device)
    material = PGNDModel(cfg)
    material.to(torch_device)
    material.load_state_dict(ckpt['material'])
    material.requires_grad_(False)
    material.eval()

    friction = torch.tensor(
        cfg.model.friction.value, device=torch_device).reshape(-1, 1)

    # ── Frozen renderer ──
    print(f'Loading renderer: {args.renderer_ckpt}')
    rdr_ckpt = torch.load(
        str(log_root / args.renderer_ckpt), map_location=torch_device)
    renderer = create_neural_mesh_renderer(
        hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
        use_vertex_colors=True, use_view_direction=True,
        device=str(torch_device),
    )
    renderer.load_state_dict(rdr_ckpt['renderer'])
    renderer.requires_grad_(False)
    renderer.eval()
    print(f'  Renderer loaded (iter={rdr_ckpt.get("iteration", "?")})')

    # ── Episodes ──
    episodes = parse_episode_range(args.episodes)
    print(f'Episodes: {len(episodes)} ({episodes[0]}-{episodes[-1]})')
    print(f'Correction: {args.correction_steps} steps, '
          f'lr={args.correction_lr}, max_d={args.max_correction}')

    output_dir = log_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(output_dir / 'eval_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Run ──
    all_metrics = []
    for episode in episodes:
        metrics = eval_episode_with_correction(
            cfg=cfg, material=material, friction=friction,
            renderer=renderer, episode=episode,
            torch_device=torch_device, wp_device=wp_device,
            log_root=log_root, output_dir=output_dir,
            correction_steps=args.correction_steps,
            correction_lr=args.correction_lr,
            max_correction=args.max_correction,
            camera_id=args.camera_id,
            save_debug=not args.no_debug,
        )
        if metrics is not None:
            all_metrics.append(metrics)

    if not all_metrics:
        print('No episodes completed!')
        return

    all_metrics = np.array(all_metrics)  # (n_episodes, n_steps, 2)

    # ── Summary ──
    print('\n' + '=' * 70)
    print('SUMMARY — Inference-Time Visual Correction')
    print('=' * 70)
    print(f'Dynamics:   {args.dynamics_ckpt}')
    print(f'Renderer:   {args.renderer_ckpt}')
    print(f'Correction: {args.correction_steps} steps @ lr={args.correction_lr}')
    print(f'Episodes:   {len(all_metrics)}')
    print()

    header = (f'{"Step":>6s}  {"Uncorrected":>12s}  {"Corrected":>12s}  '
              f'{"d(abs)":>10s}  {"d(rel)":>10s}')
    print(header)
    print('-' * 58)

    rows = []
    for s in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        if s < all_metrics.shape[1]:
            u = all_metrics[:, s, 0].mean()
            c = all_metrics[:, s, 1].mean()
            da = u - c
            dr = da / max(u, 1e-8) * 100
            print(f'{s:6d}  {u:12.4f}  {c:12.4f}  '
                  f'{da:+9.4f}  {dr:+9.1f}%')
            rows.append([s, u, c, da, dr])

    with open(str(output_dir / 'summary.txt'), 'w') as f:
        f.write(f'Dynamics: {args.dynamics_ckpt}\n')
        f.write(f'Renderer: {args.renderer_ckpt}\n')
        f.write(f'Correction: {args.correction_steps} steps, '
                f'lr={args.correction_lr}, max={args.max_correction}\n')
        f.write(f'Episodes: {len(all_metrics)}\n\n')
        f.write(header + '\n')
        for s, u, c, da, dr in rows:
            f.write(f'{s:6d}  {u:12.4f}  {c:12.4f}  '
                    f'{da:+9.4f}  {dr:+9.1f}%\n')

    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()
