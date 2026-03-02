#!/usr/bin/env python3
"""
viz_phase2.py — Visualize Phase 2 dynamics training progress.

Loads a Phase 2 checkpoint, rolls out episodes, renders with frozen renderer,
and saves 6-column comparison panels.

Layout per step:
  [GT Image | GT PC | Pred PC | Overlay | Rendered | Diff×3 + L1/SSIM]

Usage:
    cd ~/pgnd/experiments/train
    python viz_phase2.py \
        --phase2_ckpt cloth/train_v4d_phase2/ckpt/005000.pt \
        --renderer_ckpt cloth/renderer_v4d/ckpt/200000.pt \
        --episodes 610,615,620 \
        --steps 0,5,10,15,20 \
        --output_dir ~/pgnd_viz_phase2

    # Or use 'latest' to auto-find the newest checkpoint:
    python viz_phase2.py \
        --phase2_ckpt latest \
        --renderer_ckpt cloth/renderer_v4d/ckpt/200000.pt \
        --episodes 610,615,620
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup paths
root = Path(__file__).resolve().parent
sys.path.append(str(root.parent.parent))
sys.path.append(str(root.parent))

# Bypass open3d/numpy crash: register dummy pgnd module before importing submodules
import types
_pgnd_root = root.parent.parent / 'pgnd'
pgnd_dummy = types.ModuleType('pgnd')
pgnd_dummy.__path__ = [str(_pgnd_root)]
pgnd_dummy.__package__ = 'pgnd'
sys.modules['pgnd'] = pgnd_dummy

import warp as wp
from omegaconf import OmegaConf
from tqdm import trange

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import get_root
import kornia

from train.render_loss_ablation2 import (
    create_render_loss_module_ablation2,
    compute_mesh_from_particles,
    project_vertex_colors,
)
from train.render_loss import GTImageLoader, setup_camera_for_render


def transform_gripper_points(cfg, gripper_points, gripper):
    """Same as train_eval.py."""
    dx = cfg.sim.num_grids[-1]
    gripper_xyz = gripper[:, :, :, :3]
    gripper_v = gripper[:, :, :, 3:6]
    gripper_quat = gripper[:, :, :, 6:10]
    num_steps = gripper_xyz.shape[1]
    num_grippers = gripper_xyz.shape[2]
    gripper_mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(gripper_quat)
    gripper_points = gripper_points[:, None, None].repeat(1, num_steps, num_grippers, 1, 1)
    gripper_x = gripper_points @ gripper_mat + gripper_xyz[:, :, :, None]
    bsz = gripper_x.shape[0]
    num_points = gripper_x.shape[3]

    gripper_quat_vel = gripper[:, :, :, 10:13]
    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)
    gripper_v_expand = gripper_v[:, :, :, None].repeat(1, 1, 1, num_points, 1)
    gripper_points_from_axis = gripper_x - gripper_xyz[:, :, :, None]
    grid_from_gripper_axis = gripper_points_from_axis - \
        (gripper_quat_axis[:, :, :, None] * gripper_points_from_axis).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, :, None]
    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
    gripper_v = gripper_v_expand.reshape(bsz, num_steps, num_grippers * num_points, 3)
    gripper_x = gripper_x.reshape(bsz, num_steps, num_grippers * num_points, 3)

    gripper_x_mask = (gripper_x[:, :, :, 0] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 0] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 1] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 1] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 2] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 2] < 1 - (dx * (cfg.model.clip_bound + 0.5)))
    return gripper_x, gripper_v, gripper_x_mask


def draw_point_cloud(pts, h, w, color, label, cam_settings=None, coord_transform=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts

    if cam_settings is not None and coord_transform is not None:
        pts_t = torch.tensor(pts_np, dtype=torch.float32).cuda()
        pts_world = coord_transform.inverse_transform(pts_t).cpu().numpy()
        K = cam_settings['k']
        w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]
        pts_cam = (R_cam @ pts_world.T + t_cam.reshape(3, 1)).T
        pts_2d = (K @ pts_cam.T).T
        u = (pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8) / 848.0 * w).astype(int)
        v = (pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8) / 480.0 * h).astype(int)
        valid = (pts_cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_valid = pts_cam[valid, 2]
        if len(d_valid) > 0:
            d_min, d_max = d_valid.min(), d_valid.max()
            d_norm = (d_valid - d_min) / (d_max - d_min + 1e-8)
            u_v, v_v = u[valid], v[valid]
            base = np.array(color, dtype=float) / 255.0
            for i in range(len(u_v)):
                c = base * (1.0 - 0.3 * d_norm[i])
                cv2.circle(canvas, (u_v[i], v_v[i]), 2,
                           (int(c[0]*255), int(c[1]*255), int(c[2]*255)), -1)
    else:
        mins = pts_np.min(axis=0)
        maxs = pts_np.max(axis=0)
        span = (maxs - mins).max() + 1e-8
        margin = 15
        u = ((pts_np[:, 0] - mins[0]) / span * (w - 2*margin) + margin).astype(int)
        v = ((pts_np[:, 2] - mins[2]) / span * (h - 2*margin - 15) + margin + 15).astype(int)
        for i in range(len(u)):
            if 0 <= u[i] < w and 0 <= v[i] < h:
                cv2.circle(canvas, (u[i], v[i]), 2, color, -1)
    if label:
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return canvas


def draw_overlay(pred_pts, gt_pts, h, w, cam_settings=None, coord_transform=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pred_np = pred_pts.detach().cpu().numpy() if torch.is_tensor(pred_pts) else pred_pts
    gt_np = gt_pts.detach().cpu().numpy() if torch.is_tensor(gt_pts) else gt_pts

    if cam_settings is not None and coord_transform is not None:
        K = cam_settings['k']
        w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]

        def project(p):
            pt = torch.tensor(p, dtype=torch.float32).cuda()
            pw = coord_transform.inverse_transform(pt).cpu().numpy()
            pc = (R_cam @ pw.T + t_cam.reshape(3, 1)).T
            p2 = (K @ pc.T).T
            uu = (p2[:, 0] / (p2[:, 2] + 1e-8) / 848.0 * w).astype(int)
            vv = (p2[:, 1] / (p2[:, 2] + 1e-8) / 480.0 * h).astype(int)
            ok = (pc[:, 2] > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            return uu, vv, ok

        gu, gv, gok = project(gt_np)
        pu, pv, pok = project(pred_np)
        for i in range(len(gu)):
            if gok[i]:
                cv2.circle(canvas, (gu[i], gv[i]), 2, (0, 255, 255), -1)
        for i in range(len(pu)):
            if pok[i]:
                cv2.circle(canvas, (pu[i], pv[i]), 2, (0, 0, 255), -1)

    cv2.putText(canvas, 'GT+Pred Overlay', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, 'GT', (w - 70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(canvas, 'Pred', (w - 35, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return canvas


@torch.no_grad()
def render_particles(renderer, particles, gt_image, gt_mask, cam_settings, coord_transform):
    """Render particles through the neural mesh renderer. Returns (rendered_rgb, l1, ssim)."""
    mesh_data = compute_mesh_from_particles(particles, method='bpa')
    faces = mesh_data.face
    n_verts = mesh_data.pos.shape[0]
    vertices = particles[:n_verts]

    vertex_colors = project_vertex_colors(
        vertices_preproc=vertices,
        image=gt_image,
        cam_settings=cam_settings,
        coord_transform=coord_transform,
    )

    rendered_image, _ = renderer(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        cam_settings=cam_settings,
        coord_transform=coord_transform,
    )

    rendered_np = rendered_image.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    rendered_rgb = (rendered_np * 255).astype(np.uint8)

    # Compute L1 and SSIM
    gt_for_loss = gt_image
    rendered_for_loss = rendered_image
    if gt_mask is not None:
        gt_for_loss = gt_image * gt_mask
        rendered_for_loss = rendered_image * gt_mask
    l1_val = F.l1_loss(rendered_for_loss, gt_for_loss).item()

    ssim_val = 0.0
    try:
        from kornia.metrics import ssim as kornia_ssim
        ssim_map = kornia_ssim(rendered_for_loss.unsqueeze(0), gt_for_loss.unsqueeze(0), window_size=11)
        ssim_val = ssim_map.mean().item()
    except ImportError:
        try:
            from kornia.losses import ssim_loss as _ssim_loss
            ssim_val = 1.0 - _ssim_loss(rendered_for_loss.unsqueeze(0), gt_for_loss.unsqueeze(0), window_size=11).item()
        except Exception:
            pass
    except Exception:
        pass

    return rendered_rgb, l1_val, ssim_val


def make_panel(gt_rgb, gt_mask_np, pred_pts, gt_pts, rendered_rgb, l1_val, ssim_val, mde_val,
               cam_settings, coord_transform, episode, step, ckpt_name):
    """Build 6-column panel: [GT Image | GT PC | Pred PC | Overlay | Rendered | Diff×3]"""
    col_w = 280
    row_h = 240
    gap = 2
    n_cols = 6
    total_w = n_cols * col_w + (n_cols - 1) * gap
    total_h = row_h + 35
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    header = f'{ckpt_name} | ep {episode} step {step} | MDE={mde_val:.4f} | L1={l1_val:.4f} | SSIM={ssim_val:.3f}'
    cv2.putText(canvas, header, (5, 22), font, 0.55, (255, 255, 255), 1)
    y0 = 35

    def cx(i):
        return i * (col_w + gap)

    # Col 0: GT Image
    if gt_rgb is not None:
        gt_resized = cv2.resize(gt_rgb, (col_w, row_h))
        canvas[y0:y0+row_h, cx(0):cx(0)+col_w] = cv2.cvtColor(gt_resized, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'GT Image', (cx(0)+5, y0+18), font, 0.4, (200, 200, 200), 1)

    # Col 1: GT PC
    gt_pc = draw_point_cloud(gt_pts, h=row_h, w=col_w, color=(0, 255, 128), label='GT PC',
                              cam_settings=cam_settings, coord_transform=coord_transform)
    canvas[y0:y0+row_h, cx(1):cx(1)+col_w] = gt_pc

    # Col 2: Pred PC
    pred_pc = draw_point_cloud(pred_pts, h=row_h, w=col_w, color=(0, 0, 255), label='Pred PC',
                                cam_settings=cam_settings, coord_transform=coord_transform)
    canvas[y0:y0+row_h, cx(2):cx(2)+col_w] = pred_pc

    # Col 3: Overlay
    overlay = draw_overlay(pred_pts, gt_pts, h=row_h, w=col_w,
                            cam_settings=cam_settings, coord_transform=coord_transform)
    canvas[y0:y0+row_h, cx(3):cx(3)+col_w] = overlay

    # Col 4: Rendered
    if rendered_rgb is not None:
        rend_resized = cv2.resize(rendered_rgb, (col_w, row_h))
        canvas[y0:y0+row_h, cx(4):cx(4)+col_w] = cv2.cvtColor(rend_resized, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'Rendered', (cx(4)+5, y0+18), font, 0.4, (200, 200, 200), 1)

    # Col 5: Diff
    if rendered_rgb is not None and gt_rgb is not None:
        gt_r = cv2.resize(gt_rgb, (col_w, row_h))
        rend_r = cv2.resize(rendered_rgb, (col_w, row_h))
        diff = np.abs(rend_r.astype(float) - gt_r.astype(float))
        diff_amp = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
        canvas[y0:y0+row_h, cx(5):cx(5)+col_w] = cv2.cvtColor(diff_amp, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'Diff 3x', (cx(5)+5, y0+18), font, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f'L1={l1_val:.4f}  SSIM={ssim_val:.3f}',
                    (cx(5)+5, y0+row_h-12), font, 0.4, (255, 255, 100), 1)

    return canvas


def find_latest_ckpt(ckpt_dir):
    """Find the latest checkpoint in a directory."""
    ckpts = sorted(ckpt_dir.glob('*.pt'))
    ckpts = [c for c in ckpts if c.stem.isdigit()]
    if not ckpts:
        return None
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 2 dynamics training progress')
    parser.add_argument('--phase2_ckpt', type=str, required=True,
                        help='Phase 2 checkpoint (relative to log/) or "latest" for auto-detect')
    parser.add_argument('--renderer_ckpt', type=str, required=True,
                        help='Pre-trained renderer checkpoint (relative to log/)')
    parser.add_argument('--episodes', type=str, default='610,615,620',
                        help='Comma-separated episode numbers to visualize')
    parser.add_argument('--steps', type=str, default='0,5,10,15,20',
                        help='Comma-separated rollout steps to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for panels (default: log/cloth/train_v4d_phase2/viz/)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to hydra config (default: auto-detect from phase2 experiment)')
    args = parser.parse_args()

    log_root = get_root(__file__) / 'log'

    # Find checkpoint
    if args.phase2_ckpt == 'latest':
        ckpt_dir = log_root / 'cloth' / 'train_v4d_phase2' / 'ckpt'
        ckpt_path = find_latest_ckpt(ckpt_dir)
        if ckpt_path is None:
            print(f'No checkpoints found in {ckpt_dir}')
            return
        print(f'[viz] Using latest checkpoint: {ckpt_path}')
    else:
        ckpt_path = log_root / args.phase2_ckpt
    assert ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}'

    ckpt_name = ckpt_path.stem
    renderer_ckpt_path = log_root / args.renderer_ckpt
    assert renderer_ckpt_path.exists(), f'Renderer checkpoint not found: {renderer_ckpt_path}'

    episodes = [int(e) for e in args.episodes.split(',')]
    viz_steps = [int(s) for s in args.steps.split(',')]

    # Load config
    if args.config:
        config_path = args.config
    else:
        # Auto-detect from experiment directory
        exp_dir = ckpt_path.parent.parent
        config_path = str(exp_dir / 'hydra.yaml')
    assert Path(config_path).exists(), f'Config not found: {config_path}'
    cfg = OmegaConf.load(config_path)
    print(f'[viz] Config: {config_path}')

    # Setup
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    torch_device = torch.device('cuda:0')
    wp_device = wp.get_device('cuda:0')

    # Load material model
    material = PGNDModel(cfg)
    material.to(torch_device)
    ckpt_data = torch.load(str(ckpt_path), map_location=torch_device)
    material.load_state_dict(ckpt_data['material'])
    material.eval()
    print(f'[viz] Loaded dynamics from {ckpt_path}')

    friction = Friction(np.array([cfg.model.friction.value]))
    friction.to(torch_device)
    friction.eval()

    # Load renderer
    render_loss_module = create_render_loss_module_ablation2(
        cfg, log_root,
        lambda_render=0.1,
        lambda_ssim=0.2,
        render_every_n_steps=1,
        camera_id=getattr(cfg.train, 'render_camera_id', 1),
    )
    renderer_ckpt = torch.load(str(renderer_ckpt_path), map_location=torch_device)
    render_loss_module.renderer.load_state_dict(renderer_ckpt['renderer'])
    render_loss_module.renderer.eval()
    render_loss_module.renderer.requires_grad_(False)
    print(f'[viz] Loaded renderer from {renderer_ckpt_path}')

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent.parent / 'viz' / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[viz] Output: {output_dir}')

    # Load eval dataset
    source_dataset_root = log_root / str(cfg.train.source_dataset_name)

    if cfg.sim.gripper_points:
        gripper_dataset = RealGripperDataset(cfg, device=torch_device)

    # Run episodes
    for episode in episodes:
        print(f'\n[viz] Episode {episode}')
        eval_dataset = RealTeleopBatchDataset(
            cfg,
            dataset_root=log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root,
            device=torch_device,
            num_steps=cfg.sim.num_steps,
            eval_episode_name=f'episode_{episode:04d}',
        )
        from torch.utils.data import DataLoader
        dl = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        init_state, actions, gt_states, downsample_indices = next(iter(dl))

        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
        x = x.to(torch_device)
        v = v.to(torch_device)
        x_his = x_his.to(torch_device)
        v_his = v_his.to(torch_device)
        actions = actions.to(torch_device)

        if cfg.sim.gripper_points:
            gp_dl = DataLoader(gripper_dataset, batch_size=1, shuffle=False, num_workers=0)
            gripper_points, _ = next(iter(gp_dl))
            gripper_points = gripper_points.to(torch_device)
            gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, actions)

        gt_x, gt_v = gt_states
        gt_x = gt_x.to(torch_device)
        gt_v = gt_v.to(torch_device)

        batch_size = gt_x.shape[0]
        num_steps_total = gt_x.shape[1]
        num_particles = gt_x.shape[2]

        if cfg.sim.gripper_points:
            num_gripper_particles = gripper_x.shape[2]
            num_particles_orig = num_particles
            num_particles = num_particles + num_gripper_particles

        sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, wp_device, requires_grad=False)
        statics = StaticsBatch()
        statics.init(shape=(batch_size, num_particles), device=wp_device)
        statics.update_clip_bound(clip_bound)
        statics.update_enabled(enabled)
        colliders = CollidersBatch()

        num_grippers = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
        colliders.init(shape=(batch_size, num_grippers), device=wp_device)
        if num_grippers > 0:
            colliders.initialize_grippers(actions[:, 0])

        enabled = enabled.to(torch_device)

        # Setup render loss module for this episode
        ep_name = f'episode_{episode:04d}'
        render_active = render_loss_module.setup_episode(
            episode_name=ep_name,
            particles_0=x[0].detach(),
        )
        cam = render_loss_module.cam_settings
        ct = render_loss_module.coord_transform

        max_step = max(viz_steps) + 1
        max_step = min(max_step, num_steps_total)

        with torch.no_grad():
            for step in trange(max_step, desc=f'  ep {episode}'):
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
                x, v = sim(statics, colliders, step, x, v, friction.mu[None].repeat(batch_size, 1), pred)

                if cfg.sim.n_history > 0:
                    if cfg.sim.gripper_points:
                        x_his_p = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None]], dim=2)
                        v_his_p = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None]], dim=2)
                        x_his = x_his_p.reshape(batch_size, num_particles_orig, -1)
                        v_his = v_his_p.reshape(batch_size, num_particles_orig, -1)
                    else:
                        x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None]], dim=2).reshape(batch_size, num_particles, -1)
                        v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None]], dim=2).reshape(batch_size, num_particles, -1)

                if cfg.sim.gripper_points:
                    x = x[:, :num_particles_orig]
                    v = v[:, :num_particles_orig]
                    enabled = enabled[:, :num_particles_orig]

                # Generate panel at requested steps
                if step in viz_steps:
                    pred_pts = x[0].detach()
                    gt_pts = gt_x[0, step].detach()
                    mde = torch.norm(pred_pts - gt_pts, dim=-1).mean().item()

                    # Load GT image
                    gt_image = render_loss_module.gt_loader.load_frame(step)
                    gt_mask = render_loss_module.gt_loader.load_mask(step) if gt_image is not None else None

                    gt_rgb = None
                    if gt_image is not None:
                        gt_np = gt_image.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        if gt_mask is not None:
                            gt_np = gt_np * gt_mask.cpu().permute(1, 2, 0).numpy()
                        gt_rgb = (gt_np * 255).astype(np.uint8)

                    # Render
                    rendered_rgb, l1_val, ssim_val = None, 0.0, 0.0
                    if render_active and gt_image is not None:
                        try:
                            rendered_rgb, l1_val, ssim_val = render_particles(
                                render_loss_module.renderer, pred_pts.cuda(),
                                gt_image, gt_mask, cam, ct,
                            )
                        except Exception as e:
                            print(f'    Render failed at step {step}: {e}')

                    panel = make_panel(
                        gt_rgb=gt_rgb, gt_mask_np=None,
                        pred_pts=pred_pts, gt_pts=gt_pts,
                        rendered_rgb=rendered_rgb, l1_val=l1_val, ssim_val=ssim_val,
                        mde_val=mde, cam_settings=cam, coord_transform=ct,
                        episode=episode, step=step, ckpt_name=ckpt_name,
                    )

                    out_path = output_dir / f'ep{episode:04d}_step{step:03d}.jpg'
                    cv2.imwrite(str(out_path), panel)
                    print(f'    step {step}: MDE={mde:.4f} L1={l1_val:.4f} SSIM={ssim_val:.3f} → {out_path.name}')

    # Also make a summary grid: all episodes × all steps
    print(f'\n[viz] Generating summary grid...')
    all_panels = []
    for episode in episodes:
        for step in viz_steps:
            p = output_dir / f'ep{episode:04d}_step{step:03d}.jpg'
            if p.exists():
                all_panels.append(cv2.imread(str(p)))

    if all_panels:
        ph, pw = all_panels[0].shape[:2]
        n_cols_grid = len(viz_steps)
        n_rows_grid = len(episodes)
        grid = np.zeros((n_rows_grid * ph, n_cols_grid * pw, 3), dtype=np.uint8)
        for idx, panel in enumerate(all_panels):
            r = idx // n_cols_grid
            c = idx % n_cols_grid
            grid[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = panel
        grid_path = output_dir / 'summary_grid.jpg'
        cv2.imwrite(str(grid_path), grid)
        print(f'[viz] Summary grid: {grid_path}')

    print(f'\n[viz] Done! {len(episodes)} episodes × {len(viz_steps)} steps → {output_dir}')


if __name__ == '__main__':
    main()