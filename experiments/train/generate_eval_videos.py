#!/usr/bin/env python3
"""
generate_eval_videos.py — Two video types for eval visualization.

VIDEO TYPE 1: "rollout" — For each episode, 30-step rollout video.
    4 columns: GT masked image | GT point cloud | Rendered prediction | Predicted point cloud
    One row per model (baseline, phase2, visual).
    Frame = one timestep. 30 frames per episode.

VIDEO TYPE 2: "comparison" — Side-by-side model comparison across episodes.
    Same layout but stitches 5 random episodes end-to-end with episode labels.

Usage:
    cd ~/pgnd/experiments/train
    python generate_eval_videos.py --mode rollout --episodes 610 620 630
    python generate_eval_videos.py --mode comparison --num_random_episodes 5
    python generate_eval_videos.py --mode both
"""

import sys
import os
import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
from omegaconf import OmegaConf
from tqdm import tqdm, trange

# Module setup
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir.parent.parent))
sys.path.append(str(root_dir.parent))
_pgnd_root = root_dir.parent.parent / 'pgnd'
pgnd_dummy = types.ModuleType('pgnd')
pgnd_dummy.__path__ = [str(_pgnd_root)]
pgnd_dummy.__package__ = 'pgnd'
sys.modules['pgnd'] = pgnd_dummy

import warp as wp
from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import get_root
import kornia
from torch.utils.data import DataLoader

try:
    from neural_mesh_renderer import create_neural_mesh_renderer, project_vertex_colors
    from render_loss_ablation2 import PGNDCoordinateTransform
    from render_loss import GTImageLoader
    from build_cloth_mesh import compute_mesh_from_particles
    HAS_RENDERER = True
except ImportError as e:
    print(f"[warn] Neural renderer not available: {e}")
    HAS_RENDERER = False

try:
    from visual_encoder import VisualEncoder
    from pgnd_visual import PGNDVisualModel
    HAS_VISUAL = True
except ImportError:
    try:
        from train.visual_encoder import VisualEncoder
        from train.pgnd_visual import PGNDVisualModel
        HAS_VISUAL = True
    except ImportError:
        HAS_VISUAL = False

root: Path = get_root(__file__)


# =============================================================================
# Helpers (same as compare_3_ablations.py)
# =============================================================================

def transform_gripper_points(cfg, gripper_points, gripper):
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


def compute_mde(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean().item()


def find_latest_ckpt(ckpt_dir):
    ckpts = sorted(Path(ckpt_dir).glob('*.pt'))
    ckpts = [c for c in ckpts if c.stem.isdigit()]
    return ckpts[-1] if ckpts else None


# =============================================================================
# Episode setup — camera, renderer, GT loader
# =============================================================================

class EpisodeContext:
    """Loads camera calibration, renderer, and GT image loader for one episode."""

    def __init__(self, cfg, episode, device, renderer_ckpt_path=None):
        self.device = device
        self.active = False
        self.cam_settings = None
        self.coord_transform = None
        self.gt_loader = None
        self.renderer = None
        self.episode = episode

        if not HAS_RENDERER:
            return

        log_root = root / 'log'
        try:
            source_root = log_root / str(cfg.train.source_dataset_name)
            ep_path = source_root / f'episode_{episode:04d}'
            self.coord_transform = PGNDCoordinateTransform(cfg, ep_path).to_cuda()

            meta = np.loadtxt(str(ep_path / 'meta.txt'))
            with open(source_root / 'metadata.json') as f:
                metadata = json.load(f)
            entry = metadata[episode]
            rec_name = Path(entry['path']).parent.name
            src_ep = int(meta[0])
            n_hist = int(cfg.sim.n_history)
            load_skip = int(cfg.train.dataset_load_skip_frame)
            ds_skip = int(cfg.train.dataset_skip_frame)
            frame_start = int(meta[1]) + n_hist * load_skip * ds_skip

            ep_dir = log_root / 'data_cloth' / rec_name / f'episode_{src_ep:04d}'
            calib = ep_dir / 'calibration'
            intr = np.load(str(calib / 'intrinsics.npy'))
            rvec = np.load(str(calib / 'rvecs.npy'))
            tvec = np.load(str(calib / 'tvecs.npy'))
            R = cv2.Rodrigues(rvec[1])[0]
            t = tvec[1, :, 0]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            w2c = np.linalg.inv(c2w).astype(np.float32)
            self.cam_settings = {'w': 848, 'h': 480, 'k': intr[1], 'w2c': w2c}

            self.gt_loader = GTImageLoader(
                episode_dir=ep_dir, source_frame_start=frame_start,
                camera_id=1, image_size=(480, 848), skip_frame=load_skip * ds_skip)

            if renderer_ckpt_path and Path(renderer_ckpt_path).exists():
                ckpt = torch.load(str(renderer_ckpt_path), map_location=device)
                self.renderer = create_neural_mesh_renderer(
                    hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
                    use_vertex_colors=True, use_view_direction=True, device=str(device))
                self.renderer.load_state_dict(ckpt['renderer'], strict=False)
                self.renderer.eval()

            self.active = True
        except Exception as e:
            print(f'  [ctx] Episode {episode} setup failed: {e}')

    @torch.no_grad()
    def get_gt_image(self, step):
        """Returns masked GT image as (H, W, 3) uint8 RGB, or None."""
        if self.gt_loader is None:
            return None
        gt_img = self.gt_loader.load_frame(step)
        if gt_img is None:
            return None
        gt_mask = self.gt_loader.load_mask(step)
        gt_np = gt_img.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        if gt_mask is not None:
            gt_np = gt_np * gt_mask.cpu().permute(1, 2, 0).numpy()
        return (gt_np * 255).astype(np.uint8)

    @torch.no_grad()
    def render_particles(self, particles, step):
        """Render particles through neural mesh renderer. Returns (H, W, 3) uint8 RGB or None."""
        if self.renderer is None or self.gt_loader is None:
            return None
        try:
            particles = particles.to(self.device)
            gt_img = self.gt_loader.load_frame(step)
            if gt_img is None:
                return None
            mesh = compute_mesh_from_particles(particles, method='bpa')
            verts = particles[:mesh.pos.shape[0]]
            vcols = project_vertex_colors(verts, gt_img, self.cam_settings, self.coord_transform)
            rendered, _ = self.renderer(verts, mesh.face, vcols, self.cam_settings, self.coord_transform)
            rend_np = rendered.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            return (rend_np * 255).astype(np.uint8)
        except Exception:
            return None


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_pc_cam(pts, h, w, color, cam, ct):
    """Project point cloud to camera view. Returns (h, w, 3) uint8 BGR canvas."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    if cam is None or ct is None:
        return canvas
    pts_t = torch.tensor(pts_np, dtype=torch.float32).cuda()
    pw = ct.inverse_transform(pts_t).cpu().numpy()
    K, w2c = cam['k'], cam['w2c']
    pc = (w2c[:3, :3] @ pw.T + w2c[:3, 3:4]).T
    p2 = (K @ pc.T).T
    u = (p2[:, 0] / (p2[:, 2] + 1e-8) / 848 * w).astype(int)
    v = (p2[:, 1] / (p2[:, 2] + 1e-8) / 480 * h).astype(int)
    ok = (pc[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    dv = pc[ok, 2]
    if len(dv) > 0:
        dn = (dv - dv.min()) / (dv.max() - dv.min() + 1e-8)
        uv, vv = u[ok], v[ok]
        base = np.array(color, dtype=float) / 255
        for i in range(len(uv)):
            c = base * (1.0 - 0.3 * dn[i])
            cv2.circle(canvas, (uv[i], vv[i]), 2, (int(c[2]*255), int(c[1]*255), int(c[0]*255)), -1)
    return canvas


# =============================================================================
# Rollout (supports PGNDModel and PGNDVisualModel)
# =============================================================================

def setup_visual_encoder(cfg, episode, device):
    if not HAS_VISUAL or not HAS_RENDERER:
        return None, None
    log_root = root / 'log'
    try:
        source_root = log_root / str(cfg.train.source_dataset_name)
        ep_path = source_root / f'episode_{episode:04d}'
        meta = np.loadtxt(str(ep_path / 'meta.txt'))
        with open(source_root / 'metadata.json') as f:
            metadata = json.load(f)
        entry = metadata[episode]
        rec_name = Path(entry['path']).parent.name
        src_ep = int(meta[0])
        n_hist = int(cfg.sim.n_history)
        load_skip = int(cfg.train.dataset_load_skip_frame)
        ds_skip = int(cfg.train.dataset_skip_frame)
        frame_start = int(meta[1]) + n_hist * load_skip * ds_skip
        ep_dir = log_root / 'data_cloth' / rec_name / f'episode_{src_ep:04d}'
        ct = PGNDCoordinateTransform(cfg, ep_path).to_cuda()
        calib = ep_dir / 'calibration'
        intr = np.load(str(calib / 'intrinsics.npy'))
        rvec = np.load(str(calib / 'rvecs.npy'))
        tvec = np.load(str(calib / 'tvecs.npy'))
        R = cv2.Rodrigues(rvec[1])[0]
        t = tvec[1, :, 0]
        c2w = np.eye(4, dtype=np.float64); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
        w2c = np.linalg.inv(c2w).astype(np.float32)
        cam = {'w': 848, 'h': 480, 'k': intr[1], 'w2c': w2c}
        enc = VisualEncoder(model_name='dinov2_vits14', feature_dim=64,
                            camera_ids=[1], image_size=(480, 848), device=str(device))
        enc.setup_episode(coord_transform=ct, cam_settings_dict={1: cam})
        gt_loader = GTImageLoader(episode_dir=ep_dir, source_frame_start=frame_start,
                                  camera_id=1, image_size=(480, 848), skip_frame=load_skip * ds_skip)
        return enc, gt_loader
    except Exception as e:
        print(f'  [vis_enc] Setup failed ep {episode}: {e}')
        return None, None


@torch.no_grad()
def rollout_episode_full(cfg, ckpt_path, episode, device, wp_device,
                         is_visual=False, vis_dim=64):
    """Full rollout returning per-step predictions and GT."""
    log_root = root / 'log'
    source_root = log_root / str(cfg.train.source_dataset_name)

    dataset = RealTeleopBatchDataset(
        cfg, dataset_root=log_root / cfg.train.dataset_name / 'state',
        source_data_root=source_root, device=device,
        num_steps=cfg.sim.num_steps, eval_episode_name=f'episode_{episode:04d}')
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if cfg.sim.gripper_points:
        gp_ds = RealGripperDataset(cfg, device=device)
        gp_dl = DataLoader(gp_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    ckpt = torch.load(str(log_root / ckpt_path), map_location=device)

    if is_visual and HAS_VISUAL:
        material = PGNDVisualModel(cfg, vis_dim=vis_dim)
        material.to(device)
        material.load_state_dict(ckpt['material'])
        vis_enc, vis_gt_loader = setup_visual_encoder(cfg, episode, device)
        if vis_enc is not None and 'visual_proj' in ckpt:
            vis_enc.proj.load_state_dict(ckpt['visual_proj'])
    else:
        material = PGNDModel(cfg)
        material.to(device)
        material.load_state_dict(ckpt['material'])
        vis_enc, vis_gt_loader = None, None
    material.eval()

    friction = Friction(np.array([cfg.model.friction.value]))
    friction.to(device); friction.eval()

    init_state, actions, gt_states, _ = next(iter(dl))
    x, v, x_his, v_his, clip_bound, enabled, _ = init_state
    x, v = x.to(device), v.to(device)
    x_his, v_his = x_his.to(device), v_his.to(device)
    actions = actions.to(device)

    if cfg.sim.gripper_points:
        gp, _ = next(iter(gp_dl))
        gp = gp.to(device)
        gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gp, actions)

    gt_x, gt_v = gt_states
    gt_x = gt_x.to(device)
    bs = gt_x.shape[0]; num_steps = gt_x.shape[1]; num_particles = gt_x.shape[2]

    if cfg.sim.gripper_points:
        n_gp = gripper_x.shape[2]; n_orig = num_particles; num_particles += n_gp

    sim = CacheDiffSimWithFrictionBatch(cfg, num_steps, bs, wp_device, requires_grad=False)
    statics = StaticsBatch()
    statics.init(shape=(bs, num_particles), device=wp_device)
    statics.update_clip_bound(clip_bound); statics.update_enabled(enabled)
    colliders = CollidersBatch()
    n_grip = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
    colliders.init(shape=(bs, n_grip), device=wp_device)
    if n_grip > 0: colliders.initialize_grippers(actions[:, 0])
    enabled = enabled.to(device)

    preds, gts = [], []
    for step in range(num_steps):
        if n_grip > 0: colliders.update_grippers(actions[:, step])
        if cfg.sim.gripper_points:
            x = torch.cat([x, gripper_x[:, step]], dim=1)
            v = torch.cat([v, gripper_v[:, step]], dim=1)
            x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], n_gp, cfg.sim.n_history*3), device=device)], dim=1)
            v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], n_gp, cfg.sim.n_history*3), device=device)], dim=1)
            if enabled.shape[1] < num_particles:
                enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
            statics.update_enabled(enabled.cpu())

        vis_feat = None
        if vis_enc is not None and vis_gt_loader is not None:
            try:
                img = vis_gt_loader.load_frame(step)
                if img is not None:
                    x_cloth = x[:, :n_orig] if cfg.sim.gripper_points else x
                    vis_feat = vis_enc(x_cloth, {1: img})
                    if cfg.sim.gripper_points:
                        pad = torch.zeros(vis_feat.shape[0], x.shape[1] - n_orig, vis_dim, device=device)
                        vis_feat = torch.cat([vis_feat, pad], dim=1)
            except Exception:
                vis_feat = None

        if is_visual and HAS_VISUAL:
            pred = material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
        else:
            pred = material(x, v, x_his, v_his, enabled)

        x, v = sim(statics, colliders, step, x, v, friction.mu[None].repeat(bs, 1), pred)

        if cfg.sim.n_history > 0:
            if cfg.sim.gripper_points:
                xh = torch.cat([x_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:], x[:, :n_orig, None]], dim=2)
                vh = torch.cat([v_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:], v[:, :n_orig, None]], dim=2)
                x_his = xh.reshape(bs, n_orig, -1); v_his = vh.reshape(bs, n_orig, -1)
            else:
                x_his = torch.cat([x_his.reshape(bs, num_particles, -1, 3)[:, :, 1:], x[:, :, None]], dim=2).reshape(bs, num_particles, -1)
                v_his = torch.cat([v_his.reshape(bs, num_particles, -1, 3)[:, :, 1:], v[:, :, None]], dim=2).reshape(bs, num_particles, -1)

        if cfg.sim.gripper_points:
            x = x[:, :n_orig]; v = v[:, :n_orig]; enabled = enabled[:, :n_orig]

        preds.append(x[0].cpu())
        gts.append(gt_x[0, step].cpu())

    return preds, gts


# =============================================================================
# Frame builders
# =============================================================================

def build_rollout_frame(step, num_steps, episode, model_name, mde,
                        gt_rgb, gt_pts, rendered_rgb, pred_pts,
                        cam, ct, col_w=424, row_h=320):
    """
    Single frame for rollout video.
    5 columns: GT masked image | GT point cloud | Rendered prediction | Predicted point cloud | Overlay
    """
    gap = 2
    header_h = 40
    n_cols = 5
    total_w = col_w * n_cols + gap * (n_cols - 1)
    total_h = header_h + row_h
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header
    header = f'Episode {episode} | {model_name} | Step {step}/{num_steps} | MDE={mde:.4f}'
    cv2.putText(canvas, header, (10, 28), font, 0.6, (255, 255, 255), 1)

    # Progress bar
    bar_x = total_w - 210
    bar_w = 200
    bar_h = 8
    bar_y = 16
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * (step + 1) / max(num_steps, 1))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 100), -1)

    y = header_h

    # Col 0: GT masked image
    if gt_rgb is not None:
        resized = cv2.resize(gt_rgb, (col_w, row_h))
        canvas[y:y+row_h, 0:col_w] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, 'GT Image', (5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 1: GT point cloud
    x1 = col_w + gap
    gt_pc = draw_pc_cam(gt_pts, row_h, col_w, (0, 255, 128), cam, ct)
    canvas[y:y+row_h, x1:x1+col_w] = gt_pc
    cv2.putText(canvas, 'GT Cloud', (x1+5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 2: Rendered prediction
    x2 = 2 * (col_w + gap)
    if rendered_rgb is not None:
        resized = cv2.resize(rendered_rgb, (col_w, row_h))
        canvas[y:y+row_h, x2:x2+col_w] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, 'Rendered Pred', (x2+5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 3: Predicted point cloud
    x3 = 3 * (col_w + gap)
    pred_pc = draw_pc_cam(pred_pts, row_h, col_w, (0, 100, 255), cam, ct)
    canvas[y:y+row_h, x3:x3+col_w] = pred_pc
    cv2.putText(canvas, 'Pred Cloud', (x3+5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 4: Overlay (GT cyan + Pred red)
    x4 = 4 * (col_w + gap)
    overlay = draw_pc_cam(gt_pts, row_h, col_w, (0, 255, 255), cam, ct)
    pred_layer = draw_pc_cam(pred_pts, row_h, col_w, (0, 0, 255), cam, ct)
    mask = pred_layer.any(axis=2)
    overlay[mask] = pred_layer[mask]
    canvas[y:y+row_h, x4:x4+col_w] = overlay
    cv2.putText(canvas, 'Overlay (GT+Pred)', (x4+5, y+18), font, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, f'MDE={mde:.4f}', (x4+5, y+row_h-10), font, 0.45, (255, 255, 100), 1)

    return canvas


def build_comparison_frame(step, num_steps, episode, models_data, ep_ctx,
                           col_w=424, row_h=240):
    """
    Single frame for comparison video.
    Header row: GT image | GT cloud
    Per-model row: Rendered pred | Pred cloud | Overlay
    """
    n_models = len(models_data)
    gap = 2
    header_h = 35
    gt_row_h = row_h
    model_row_h = row_h
    label_h = 22
    total_w = col_w * 3 + gap * 2
    total_h = header_h + gt_row_h + n_models * (label_h + model_row_h) + 10
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = ep_ctx.cam_settings
    ct = ep_ctx.coord_transform

    # Header
    cv2.putText(canvas, f'Episode {episode}  Step {step}/{num_steps}',
                (10, 25), font, 0.6, (255, 255, 255), 1)
    # Progress bar
    bar_x = total_w - 210; bar_w = 200; bar_h = 8; bar_y = 14
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * (step + 1) / max(num_steps, 1))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 100), -1)

    y = header_h

    # GT row
    gt_rgb = ep_ctx.get_gt_image(step)
    first_data = list(models_data.values())[0]
    gt_pts = first_data['gt'][step]

    if gt_rgb is not None:
        canvas[y:y+gt_row_h, 0:col_w] = cv2.cvtColor(cv2.resize(gt_rgb, (col_w, gt_row_h)), cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, 'GT Image', (5, y+18), font, 0.4, (200, 200, 200), 1)

    x1 = col_w + gap
    canvas[y:y+gt_row_h, x1:x1+col_w] = draw_pc_cam(gt_pts, gt_row_h, col_w, (0, 255, 128), cam, ct)
    cv2.putText(canvas, 'GT Cloud', (x1+5, y+18), font, 0.4, (200, 200, 200), 1)

    y += gt_row_h

    # Model colors
    model_colors = {}
    for name in models_data:
        if 'baseline' in name: model_colors[name] = (255, 180, 0)     # blue-ish in BGR
        elif 'phase2' in name: model_colors[name] = (0, 100, 255)     # red
        elif 'visual' in name: model_colors[name] = (0, 220, 0)       # green
        else: model_colors[name] = (200, 200, 200)

    for name, data in models_data.items():
        pred_pts = data['pred'][step]
        gt_m = data['gt'][step]
        mde = compute_mde(pred_pts, gt_m)
        color = model_colors.get(name, (200, 200, 200))

        # Label
        cv2.putText(canvas, f'{name}  MDE={mde:.4f}', (5, y + 16), font, 0.45, color, 1)
        y += label_h

        # Col 0: Rendered prediction
        rendered = ep_ctx.render_particles(pred_pts, step)
        if rendered is not None:
            canvas[y:y+model_row_h, 0:col_w] = cv2.cvtColor(cv2.resize(rendered, (col_w, model_row_h)), cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'Rendered', (5, y+18), font, 0.35, (180, 180, 180), 1)

        # Col 1: Predicted cloud
        x1 = col_w + gap
        canvas[y:y+model_row_h, x1:x1+col_w] = draw_pc_cam(pred_pts, model_row_h, col_w, (0, 100, 255), cam, ct)
        cv2.putText(canvas, 'Pred Cloud', (x1+5, y+18), font, 0.35, (180, 180, 180), 1)

        # Col 2: Overlay (GT cyan + Pred red)
        x2 = 2 * (col_w + gap)
        overlay = draw_pc_cam(gt_m, model_row_h, col_w, (0, 255, 255), cam, ct)
        pred_layer = draw_pc_cam(pred_pts, model_row_h, col_w, (0, 0, 255), cam, ct)
        mask = pred_layer.any(axis=2)
        overlay[mask] = pred_layer[mask]
        canvas[y:y+model_row_h, x2:x2+col_w] = overlay
        cv2.putText(canvas, 'Overlay', (x2+5, y+18), font, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, f'MDE={mde:.4f}', (x2+5, y+model_row_h-10), font, 0.4, (255, 255, 100), 1)

        y += model_row_h + 2

    # Combined overlay panel: GT (white) + all models in their colors on one wide canvas
    overlay_h = row_h
    overlay_w = col_w * 3 + gap * 2  # full width
    # Expand canvas to fit
    combined = np.zeros((y + label_h + overlay_h + 10, total_w, 3), dtype=np.uint8)
    combined[:y, :, :] = canvas[:y, :, :]
    canvas = combined

    cv2.putText(canvas, 'ALL MODELS OVERLAY', (5, y + 16), font, 0.5, (255, 255, 255), 1)
    # Legend
    legend_x = 220
    for name in models_data:
        color = model_colors.get(name, (200, 200, 200))
        cv2.circle(canvas, (legend_x, y + 11), 5, color, -1)
        cv2.putText(canvas, name, (legend_x + 10, y + 16), font, 0.35, color, 1)
        legend_x += len(name) * 8 + 30
    cv2.circle(canvas, (legend_x, y + 11), 5, (255, 255, 255), -1)
    cv2.putText(canvas, 'GT', (legend_x + 10, y + 16), font, 0.35, (255, 255, 255), 1)
    y += label_h

    # Draw GT points first (white, smaller)
    overlay_canvas = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
    if cam is not None and ct is not None:
        gt_np = gt_pts.detach().cpu().numpy() if torch.is_tensor(gt_pts) else gt_pts
        pts_t = torch.tensor(gt_np, dtype=torch.float32).cuda()
        pw = ct.inverse_transform(pts_t).cpu().numpy()
        K, w2c_m = cam['k'], cam['w2c']
        pc = (w2c_m[:3, :3] @ pw.T + w2c_m[:3, 3:4]).T
        p2 = (K @ pc.T).T
        u_gt = (p2[:, 0] / (p2[:, 2] + 1e-8) / 848 * overlay_w).astype(int)
        v_gt = (p2[:, 1] / (p2[:, 2] + 1e-8) / 480 * overlay_h).astype(int)
        ok_gt = (pc[:, 2] > 0) & (u_gt >= 0) & (u_gt < overlay_w) & (v_gt >= 0) & (v_gt < overlay_h)
        for i in range(len(u_gt)):
            if ok_gt[i]:
                cv2.circle(overlay_canvas, (u_gt[i], v_gt[i]), 2, (255, 255, 255), -1)

        # Draw each model's predictions
        # BGR colors for cv2
        pc_colors = {}
        for name in models_data:
            if 'baseline' in name: pc_colors[name] = (255, 180, 0)
            elif 'phase2' in name: pc_colors[name] = (0, 100, 255)
            elif 'visual' in name: pc_colors[name] = (0, 220, 0)
            else: pc_colors[name] = (200, 200, 200)

        for name, data in models_data.items():
            pred_pts = data['pred'][step]
            pred_np = pred_pts.detach().cpu().numpy() if torch.is_tensor(pred_pts) else pred_pts
            pts_t = torch.tensor(pred_np, dtype=torch.float32).cuda()
            pw = ct.inverse_transform(pts_t).cpu().numpy()
            pc = (w2c_m[:3, :3] @ pw.T + w2c_m[:3, 3:4]).T
            p2 = (K @ pc.T).T
            u_p = (p2[:, 0] / (p2[:, 2] + 1e-8) / 848 * overlay_w).astype(int)
            v_p = (p2[:, 1] / (p2[:, 2] + 1e-8) / 480 * overlay_h).astype(int)
            ok_p = (pc[:, 2] > 0) & (u_p >= 0) & (u_p < overlay_w) & (v_p >= 0) & (v_p < overlay_h)
            color = pc_colors.get(name, (200, 200, 200))
            for i in range(len(u_p)):
                if ok_p[i]:
                    cv2.circle(overlay_canvas, (u_p[i], v_p[i]), 2, color, -1)

    canvas[y:y+overlay_h, 0:overlay_w] = overlay_canvas
    y += overlay_h + 5

    return canvas[:y]


# =============================================================================
# Video writers
# =============================================================================

def write_video(frames, output_path, fps=5):
    if not frames:
        print(f'  No frames for {output_path}')
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    print(f'  Saved: {output_path} ({len(frames)} frames, {w}x{h})')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='both', choices=['rollout', 'comparison', 'both'])
    parser.add_argument('--baseline_ckpt', default='cloth/train/ckpt/100000.pt')
    parser.add_argument('--phase2_ckpt', default='cloth/train_v4d_phase2/ckpt/040000.pt')
    parser.add_argument('--visual_ckpt', default='latest')
    parser.add_argument('--vis_dim', type=int, default=64)
    parser.add_argument('--renderer_ckpt', default='cloth/renderer_v4d/ckpt/200000.pt')
    parser.add_argument('--episodes', nargs='+', type=int, default=None,
                        help='Specific episodes. If None, picks random from eval range.')
    parser.add_argument('--num_random_episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=35)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    log_root = root / 'log'

    # Resolve visual ckpt
    if args.visual_ckpt == 'latest':
        ckpt_dir = log_root / 'cloth' / 'train_visual_v1' / 'ckpt'
        vis_path = find_latest_ckpt(ckpt_dir)
        if vis_path:
            args.visual_ckpt = str(vis_path.relative_to(log_root))
            print(f'[visual] Latest: {args.visual_ckpt}')
        else:
            args.visual_ckpt = None
            print('[visual] No checkpoints found')

    # Models: name -> (ckpt, is_visual)
    models = {}
    if args.baseline_ckpt and (log_root / args.baseline_ckpt).exists():
        models['baseline_100k'] = (args.baseline_ckpt, False)
    if args.phase2_ckpt and (log_root / args.phase2_ckpt).exists():
        p2_name = Path(args.phase2_ckpt).stem
        models[f'phase2_{p2_name}'] = (args.phase2_ckpt, False)
    if args.visual_ckpt and (log_root / args.visual_ckpt).exists():
        vis_name = Path(args.visual_ckpt).stem
        models[f'visual_{vis_name}'] = (args.visual_ckpt, True)

    print(f'\nModels: {list(models.keys())}')

    # Config
    config_path = log_root / 'cloth' / 'train' / 'hydra.yaml'
    cfg = OmegaConf.create(yaml.load(open(config_path), Loader=yaml.CLoader))
    cfg.sim.num_steps = args.max_steps
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True

    # Episodes
    if args.episodes:
        episodes = args.episodes
    else:
        eval_start = cfg.train.eval_start_episode
        eval_end = cfg.train.eval_end_episode
        pool = list(range(eval_start, eval_end))
        episodes = sorted(random.sample(pool, min(args.num_random_episodes, len(pool))))
    print(f'Episodes: {episodes}')

    # Output
    timestamp = time.strftime("%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = str(log_root / 'eval_videos' / timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output: {args.output_dir}')

    # Init
    wp.init(); wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    device = torch.device('cuda:0')
    wp_dev = wp.get_device('cuda:0')

    renderer_path = log_root / args.renderer_ckpt if args.renderer_ckpt else None

    # =========================================================================
    # VIDEO TYPE 1: Rollout videos (one per model per episode)
    # =========================================================================
    if args.mode in ('rollout', 'both'):
        print(f'\n{"="*60}\nGenerating ROLLOUT videos\n{"="*60}')

        for episode in episodes:
            print(f'\nEpisode {episode}:')
            ep_ctx = EpisodeContext(cfg, episode, device, renderer_path)

            for model_name, (ckpt, is_visual) in models.items():
                print(f'  {model_name}...')
                preds, gts = rollout_episode_full(
                    cfg, ckpt, episode, device, wp_dev,
                    is_visual=is_visual, vis_dim=args.vis_dim)

                frames = []
                num_steps = len(preds)
                for step in tqdm(range(num_steps), desc=f'    frames', leave=False):
                    mde = compute_mde(preds[step], gts[step])
                    gt_rgb = ep_ctx.get_gt_image(step)
                    rendered_rgb = ep_ctx.render_particles(preds[step], step) if ep_ctx.active else None

                    frame = build_rollout_frame(
                        step, num_steps, episode, model_name, mde,
                        gt_rgb, gts[step], rendered_rgb, preds[step],
                        ep_ctx.cam_settings, ep_ctx.coord_transform)
                    frames.append(frame)

                video_path = os.path.join(args.output_dir, f'rollout_ep{episode:04d}_{model_name}.mp4')
                write_video(frames, video_path, fps=args.fps)

    # =========================================================================
    # VIDEO TYPE 2: Comparison video (all models side by side, all episodes)
    # =========================================================================
    if args.mode in ('comparison', 'both'):
        print(f'\n{"="*60}\nGenerating COMPARISON video\n{"="*60}')

        all_frames = []
        for episode in episodes:
            print(f'\nEpisode {episode}:')
            ep_ctx = EpisodeContext(cfg, episode, device, renderer_path)

            # Rollout all models
            ep_data = {}
            for model_name, (ckpt, is_visual) in models.items():
                print(f'  Rolling out {model_name}...')
                preds, gts = rollout_episode_full(
                    cfg, ckpt, episode, device, wp_dev,
                    is_visual=is_visual, vis_dim=args.vis_dim)
                ep_data[model_name] = {'pred': preds, 'gt': gts}

            num_steps = min(len(d['pred']) for d in ep_data.values())

            # Episode title card (2 seconds)
            if all_frames:
                title_h, title_w = all_frames[0].shape[:2]
            else:
                title_h, title_w = 800, 1280
            title_card = np.zeros((title_h, title_w, 3), dtype=np.uint8)
            cv2.putText(title_card, f'Episode {episode}',
                        (title_w // 2 - 150, title_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            for _ in range(args.fps * 2):  # 2 seconds
                all_frames.append(title_card)

            # Per-step frames
            for step in tqdm(range(num_steps), desc=f'  comparison frames', leave=False):
                frame = build_comparison_frame(
                    step, num_steps, episode, ep_data, ep_ctx)
                all_frames.append(frame)

        video_path = os.path.join(args.output_dir, 'comparison_all_episodes.mp4')
        write_video(all_frames, video_path, fps=args.fps)

    print(f'\nDone! Results in {args.output_dir}')
    print(f'  ls {args.output_dir}/')


if __name__ == '__main__':
    main()