#!/usr/bin/env python3
"""
compare_phase2.py — Compare baseline vs Phase 2 dynamics checkpoint.

Focused comparison to diagnose when and how predictions diverge.
Dense step coverage around early steps (1,2,3,5,7,10,15,20,30).

Usage:
    cd ~/pgnd/experiments/train
    python compare_phase2.py
    python compare_phase2.py --phase2_ckpt cloth/train_v4d_phase2/ckpt/015000.pt
    python compare_phase2.py --phase2_ckpt latest --episodes 610 615 620 625 630
"""

import sys
import os
import argparse
import json
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
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import trange

# Bypass open3d/numpy crash
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

from scipy.optimize import linear_sum_assignment

try:
    from neural_mesh_renderer import create_neural_mesh_renderer, project_vertex_colors
    from render_loss_ablation2 import PGNDCoordinateTransform
    from render_loss import GTImageLoader
    from build_cloth_mesh import compute_mesh_from_particles
    HAS_RENDERER = True
except ImportError as e:
    print(f"[warn] Neural renderer not available: {e}")
    HAS_RENDERER = False

root: Path = get_root(__file__)


# =============================================================================
# Helpers
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

def compute_chamfer(pred, gt, n=2000):
    if pred.shape[0] > n:
        pred = pred[torch.randperm(pred.shape[0])[:n]]
    if gt.shape[0] > n:
        gt = gt[torch.randperm(gt.shape[0])[:n]]
    d = torch.norm(pred.unsqueeze(1) - gt.unsqueeze(0), dim=-1)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2

def compute_emd(pred, gt, n=1000):
    if pred.shape[0] > n:
        pred = pred[torch.randperm(pred.shape[0])[:n]]
    if gt.shape[0] > n:
        gt = gt[torch.randperm(gt.shape[0])[:n]]
    k = min(pred.shape[0], gt.shape[0])
    cost = np.linalg.norm(pred[:k].cpu().numpy()[:, None] - gt[:k].cpu().numpy()[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return cost[r, c].mean()


def find_latest_ckpt(ckpt_dir):
    ckpts = sorted(Path(ckpt_dir).glob('*.pt'))
    ckpts = [c for c in ckpts if c.stem.isdigit()]
    return ckpts[-1] if ckpts else None


# =============================================================================
# Rollout
# =============================================================================

@torch.no_grad()
def rollout_episode(cfg, ckpt_path, episode, device, wp_device):
    """Roll out one episode, return per-step pred and gt positions."""
    log_root = root / 'log'
    source_root = log_root / str(cfg.train.source_dataset_name)

    dataset = RealTeleopBatchDataset(
        cfg, dataset_root=log_root / cfg.train.dataset_name / 'state',
        source_data_root=source_root, device=device,
        num_steps=cfg.sim.num_steps, eval_episode_name=f'episode_{episode:04d}',
    )
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if cfg.sim.gripper_points:
        gp_ds = RealGripperDataset(cfg, device=device)
        gp_dl = DataLoader(gp_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    ckpt = torch.load(str(log_root / ckpt_path), map_location=device)
    material = PGNDModel(cfg)
    material.to(device)
    material.load_state_dict(ckpt['material'])
    material.eval()

    friction = Friction(np.array([cfg.model.friction.value]))
    friction.to(device)
    friction.eval()

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
    bs = gt_x.shape[0]
    num_steps = gt_x.shape[1]
    num_particles = gt_x.shape[2]

    if cfg.sim.gripper_points:
        n_gp = gripper_x.shape[2]
        n_orig = num_particles
        num_particles += n_gp

    sim = CacheDiffSimWithFrictionBatch(cfg, num_steps, bs, wp_device, requires_grad=False)
    statics = StaticsBatch()
    statics.init(shape=(bs, num_particles), device=wp_device)
    statics.update_clip_bound(clip_bound)
    statics.update_enabled(enabled)
    colliders = CollidersBatch()
    n_grip = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
    colliders.init(shape=(bs, n_grip), device=wp_device)
    if n_grip > 0:
        colliders.initialize_grippers(actions[:, 0])
    enabled = enabled.to(device)

    preds, gts = [], []
    for step in range(num_steps):
        if n_grip > 0:
            colliders.update_grippers(actions[:, step])
        if cfg.sim.gripper_points:
            x = torch.cat([x, gripper_x[:, step]], dim=1)
            v = torch.cat([v, gripper_v[:, step]], dim=1)
            x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], n_gp, cfg.sim.n_history*3), device=device)], dim=1)
            v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], n_gp, cfg.sim.n_history*3), device=device)], dim=1)
            if enabled.shape[1] < num_particles:
                enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
            statics.update_enabled(enabled.cpu())

        pred = material(x, v, x_his, v_his, enabled)
        x, v = sim(statics, colliders, step, x, v, friction.mu[None].repeat(bs, 1), pred)

        if cfg.sim.n_history > 0:
            if cfg.sim.gripper_points:
                xh = torch.cat([x_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:], x[:, :n_orig, None]], dim=2)
                vh = torch.cat([v_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:], v[:, :n_orig, None]], dim=2)
                x_his = xh.reshape(bs, n_orig, -1)
                v_his = vh.reshape(bs, n_orig, -1)
            else:
                x_his = torch.cat([x_his.reshape(bs, num_particles, -1, 3)[:, :, 1:], x[:, :, None]], dim=2).reshape(bs, num_particles, -1)
                v_his = torch.cat([v_his.reshape(bs, num_particles, -1, 3)[:, :, 1:], v[:, :, None]], dim=2).reshape(bs, num_particles, -1)

        if cfg.sim.gripper_points:
            x = x[:, :n_orig]
            v = v[:, :n_orig]
            enabled = enabled[:, :n_orig]

        preds.append(x[0].cpu())
        gts.append(gt_x[0, step].cpu())

    return preds, gts


# =============================================================================
# Renderer wrapper
# =============================================================================

class NeuralRendererWrapper:
    """Load neural mesh renderer and render particles for any model."""

    def __init__(self, renderer_ckpt_path, cfg, episode, device):
        self.device = device
        self.active = False
        if not HAS_RENDERER:
            return

        log_root = root / 'log'
        try:
            ckpt = torch.load(str(renderer_ckpt_path), map_location=device)
            self.renderer = create_neural_mesh_renderer(
                hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
                use_vertex_colors=True, use_view_direction=True, device=str(device),
            )
            self.renderer.load_state_dict(ckpt['renderer'], strict=False)
            self.renderer.eval()

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

            self.episode_dir = log_root / 'data_cloth' / rec_name / f'episode_{src_ep:04d}'
            calib = self.episode_dir / 'calibration'
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
                episode_dir=self.episode_dir, source_frame_start=frame_start,
                camera_id=1, image_size=(480, 848), skip_frame=load_skip * ds_skip,
            )
            self.active = True
        except Exception as e:
            print(f'  [renderer] Init failed: {e}')

    @torch.no_grad()
    def render(self, particles, step):
        """Returns (rendered_rgb, gt_rgb, l1, ssim) or Nones."""
        if not self.active:
            return None, None, 0, 0
        try:
            particles = particles.to(self.device)
            gt_img = self.gt_loader.load_frame(step)
            if gt_img is None:
                return None, None, 0, 0
            gt_mask = self.gt_loader.load_mask(step)

            mesh = compute_mesh_from_particles(particles, method='bpa')
            verts = particles[:mesh.pos.shape[0]]
            vcols = project_vertex_colors(verts, gt_img, self.cam_settings, self.coord_transform)
            rendered, _ = self.renderer(verts, mesh.face, vcols, self.cam_settings, self.coord_transform)

            # GT masked
            gt_np = gt_img.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            if gt_mask is not None:
                gt_np = gt_np * gt_mask.cpu().permute(1, 2, 0).numpy()
            gt_rgb = (gt_np * 255).astype(np.uint8)

            rend_np = rendered.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            rend_rgb = (rend_np * 255).astype(np.uint8)

            # L1 / SSIM
            gt_t = gt_img if gt_mask is None else gt_img * gt_mask
            rend_t = rendered if gt_mask is None else rendered * gt_mask
            l1 = F.l1_loss(rend_t, gt_t).item()
            try:
                from kornia.metrics import ssim as ks
                ssim = ks(rend_t.unsqueeze(0), gt_t.unsqueeze(0), window_size=11).mean().item()
            except Exception:
                ssim = 0.0

            return rend_rgb, gt_rgb, l1, ssim
        except Exception as e:
            return None, None, 0, 0


# =============================================================================
# Drawing
# =============================================================================

def draw_pc(pts, h, w, color, label, cam=None, ct=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    if cam is not None and ct is not None:
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
                cv2.circle(canvas, (uv[i], vv[i]), 2, (int(c[0]*255), int(c[1]*255), int(c[2]*255)), -1)
    if label:
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return canvas


def draw_overlay(pred, gt, h, w, cam=None, ct=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if cam is None or ct is None:
        return canvas
    K, w2c = cam['k'], cam['w2c']
    R, t = w2c[:3, :3], w2c[:3, 3:4]
    def proj(p):
        pt = torch.tensor(p.detach().cpu().numpy() if torch.is_tensor(p) else p, dtype=torch.float32).cuda()
        pw = ct.inverse_transform(pt).cpu().numpy()
        pc = (R @ pw.T + t).T
        p2 = (K @ pc.T).T
        u = (p2[:, 0] / (p2[:, 2]+1e-8) / 848 * w).astype(int)
        v = (p2[:, 1] / (p2[:, 2]+1e-8) / 480 * h).astype(int)
        ok = (pc[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u, v, ok
    gu, gv, gok = proj(gt)
    pu, pv, pok = proj(pred)
    for i in range(len(gu)):
        if gok[i]: cv2.circle(canvas, (gu[i], gv[i]), 2, (0, 255, 255), -1)
    for i in range(len(pu)):
        if pok[i]: cv2.circle(canvas, (pu[i], pv[i]), 2, (0, 0, 255), -1)
    cv2.putText(canvas, 'Overlay', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, 'GT', (w-60, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(canvas, 'Pred', (w-30, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return canvas


# =============================================================================
# Panel builder
# =============================================================================

def make_panel(models_data, step, episode, renderer, viz_w=848):
    """
    Build comparison panel.

    Layout:
      Header row:  [GT Image | GT PC]
      Per model:   [Pred PC | Overlay | Rendered | Diff×3 + metrics]

    All columns camera-aligned.
    """
    n_models = len(models_data)
    col_w = viz_w // 4
    row_h = 240
    gap = 2
    total_w = col_w * 4 + gap * 3
    total_h = 30 + row_h + n_models * (row_h + 25) + 10
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = renderer.cam_settings if renderer and renderer.active else None
    ct = renderer.coord_transform if renderer and renderer.active else None

    # Get GT from first model
    first = list(models_data.values())[0]
    gt_pts = first['gt'][step]

    # Header
    cv2.putText(canvas, f'Episode {episode}  Step {step}', (5, 22), font, 0.6, (255, 255, 255), 1)
    y = 30

    # GT row: [GT Image | GT PC | (empty) | (empty)]
    gt_rgb = None
    if renderer and renderer.active:
        gt_img = renderer.gt_loader.load_frame(step)
        gt_mask = renderer.gt_loader.load_mask(step)
        if gt_img is not None:
            gn = gt_img.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            if gt_mask is not None:
                gn = gn * gt_mask.cpu().permute(1, 2, 0).numpy()
            gt_rgb = (gn * 255).astype(np.uint8)
            canvas[y:y+row_h, 0:col_w] = cv2.cvtColor(cv2.resize(gt_rgb, (col_w, row_h)), cv2.COLOR_RGB2BGR)
            cv2.putText(canvas, f'GT Image t={step}', (5, y+18), font, 0.4, (200, 200, 200), 1)

    gt_pc_img = draw_pc(gt_pts, row_h, col_w, (0, 255, 128), f'GT PC t={step}', cam, ct)
    canvas[y:y+row_h, col_w+gap:2*col_w+gap] = gt_pc_img
    y += row_h + 5

    # Per-model rows
    for name, data in models_data.items():
        pred_pts = data['pred'][step]
        gt_m = data['gt'][step]
        mde = compute_mde(pred_pts, gt_m)
        cd = compute_chamfer(pred_pts, gt_m)

        # Model label + metrics
        cv2.putText(canvas, f'{name}  MDE={mde:.4f}  CD={cd:.4f}', (5, y+15), font, 0.45, (0, 255, 255), 1)
        y += 20

        # Col 0: Pred PC
        canvas[y:y+row_h, 0:col_w] = draw_pc(pred_pts, row_h, col_w, (0, 0, 255), 'Pred PC', cam, ct)

        # Col 1: Overlay
        canvas[y:y+row_h, col_w+gap:2*col_w+gap] = draw_overlay(pred_pts, gt_m, row_h, col_w, cam, ct)

        # Col 2-3: Rendered + Diff
        if renderer and renderer.active:
            rend_rgb, _, l1, ssim = renderer.render(pred_pts, step)
            if rend_rgb is not None:
                rr = cv2.resize(rend_rgb, (col_w, row_h))
                canvas[y:y+row_h, 2*col_w+2*gap:3*col_w+2*gap] = cv2.cvtColor(rr, cv2.COLOR_RGB2BGR)
                cv2.putText(canvas, 'Rendered', (2*col_w+2*gap+5, y+18), font, 0.4, (200, 200, 200), 1)

                if gt_rgb is not None:
                    gr = cv2.resize(gt_rgb, (col_w, row_h))
                    diff = np.abs(rr.astype(float) - gr.astype(float))
                    diff_amp = np.clip(diff * 3, 0, 255).astype(np.uint8)
                    canvas[y:y+row_h, 3*col_w+3*gap:4*col_w+3*gap] = cv2.cvtColor(diff_amp, cv2.COLOR_RGB2BGR)
                    cv2.putText(canvas, f'Diff 3x  L1={l1:.4f} SSIM={ssim:.3f}',
                                (3*col_w+3*gap+5, y+18), font, 0.35, (200, 200, 200), 1)

        y += row_h + 5

    return canvas[:y]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', default='cloth/train/ckpt/100000.pt')
    parser.add_argument('--phase2_ckpt', default='latest',
                        help='Phase 2 ckpt relative to log/, or "latest"')
    parser.add_argument('--renderer_ckpt', default='cloth/renderer_v4d/ckpt/200000.pt')
    parser.add_argument('--episodes', nargs='+', type=int, default=[610, 615, 620])
    parser.add_argument('--viz_steps', nargs='+', type=int,
                        default=[1, 2, 3, 5, 7, 10, 15, 20, 30],
                        help='Dense steps for diagnosis')
    parser.add_argument('--max_steps', type=int, default=35)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    log_root = root / 'log'

    # Resolve phase2 checkpoint
    if args.phase2_ckpt == 'latest':
        ckpt_dir = log_root / 'cloth' / 'train_v4d_phase2' / 'ckpt'
        p2_path = find_latest_ckpt(ckpt_dir)
        if p2_path is None:
            print('No Phase 2 checkpoints found')
            return
        args.phase2_ckpt = str(p2_path.relative_to(log_root))
        print(f'[phase2] Latest: {args.phase2_ckpt}')

    p2_name = Path(args.phase2_ckpt).stem  # e.g. "015000"

    models = {
        'baseline_100k': args.baseline_ckpt,
        f'phase2_{p2_name}': args.phase2_ckpt,
    }

    # Load config
    config_path = log_root / 'cloth' / 'train' / 'hydra.yaml'
    if not config_path.exists():
        config_path = log_root / 'cloth' / 'train_v4d_phase2' / 'hydra.yaml'
    cfg = OmegaConf.create(yaml.load(open(config_path), Loader=yaml.CLoader))
    cfg.sim.num_steps = args.max_steps
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True

    # Output
    if args.output_dir is None:
        args.output_dir = str(log_root / 'phase2_comparison' / f'{p2_name}_{time.strftime("%H%M%S")}')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output: {args.output_dir}')

    # Init
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    device = torch.device('cuda:0')
    wp_dev = wp.get_device('cuda:0')

    # Verify checkpoints
    for name, ckpt in models.items():
        p = log_root / ckpt
        status = '✓' if p.exists() else '✗ MISSING'
        print(f'  {name}: {p} {status}')

    # All-episode metrics accumulator
    all_metrics = {name: defaultdict(list) for name in models}

    for episode in args.episodes:
        print(f'\n{"="*60}\nEpisode {episode}\n{"="*60}')

        # Init renderer for this episode
        renderer = NeuralRendererWrapper(
            log_root / args.renderer_ckpt, cfg, episode, device
        ) if HAS_RENDERER else None

        ep_data = {}
        for name, ckpt in models.items():
            if not (log_root / ckpt).exists():
                continue
            print(f'  Rolling out {name}...')
            preds, gts = rollout_episode(cfg, ckpt, episode, device, wp_dev)
            ep_data[name] = {'pred': preds, 'gt': gts}

            # Compute metrics at every step
            for s in range(len(preds)):
                mde = compute_mde(preds[s], gts[s])
                cd = compute_chamfer(preds[s], gts[s])
                emd = compute_emd(preds[s], gts[s])
                all_metrics[name][s].append({'mde': mde, 'chamfer': cd, 'emd': emd})

        # Save panels at viz_steps
        ep_dir = os.path.join(args.output_dir, f'ep{episode:04d}')
        os.makedirs(ep_dir, exist_ok=True)

        for step in args.viz_steps:
            if step >= args.max_steps:
                continue
            if not ep_data:
                continue
            max_avail = min(len(d['pred']) for d in ep_data.values())
            if step >= max_avail:
                continue

            panel = make_panel(ep_data, step, episode, renderer)
            path = os.path.join(ep_dir, f'step{step:03d}.jpg')
            cv2.imwrite(path, panel)
            # Print metrics for this step
            for name in ep_data:
                mde = compute_mde(ep_data[name]['pred'][step], ep_data[name]['gt'][step])
                print(f'    step {step:2d} {name}: MDE={mde:.4f}')

    # =========================================================================
    # Aggregate metric curves
    # =========================================================================
    print(f'\n{"="*60}\nAggregate Metrics (mean over {len(args.episodes)} episodes)\n{"="*60}')

    max_step = max(max(steps.keys()) for steps in all_metrics.values() if steps) + 1
    colors = {'baseline_100k': 'tab:blue', f'phase2_{p2_name}': 'tab:red'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_keys = ['mde', 'chamfer', 'emd']
    metric_titles = ['MDE (↓)', 'Chamfer Distance (↓)', 'EMD (↓)']

    for ax, key, title in zip(axes, metric_keys, metric_titles):
        for name in models:
            steps_data = all_metrics[name]
            xs, ys, stds = [], [], []
            for s in sorted(steps_data.keys()):
                vals = [m[key] for m in steps_data[s]]
                xs.append(s)
                ys.append(np.mean(vals))
                stds.append(np.std(vals))
            xs, ys, stds = np.array(xs), np.array(ys), np.array(stds)
            c = colors.get(name, 'tab:green')
            ax.plot(xs, ys, label=name, color=c, linewidth=2)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=c)

        ax.set_xlabel('Rollout Step')
        ax.set_ylabel(key.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Mark the ~5 step divergence zone
        ax.axvspan(3, 7, alpha=0.05, color='red')
        ax.axvline(5, color='gray', linestyle='--', alpha=0.3, label='~step 5')

    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, 'metric_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f'Saved: {curve_path}')

    # Print table
    print(f'\n{"Step":>5}', end='')
    for name in models:
        print(f'  {name:>20}', end='')
    print()
    for s in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        if s >= max_step:
            break
        print(f'{s:>5}', end='')
        for name in models:
            if s in all_metrics[name]:
                mde = np.mean([m['mde'] for m in all_metrics[name][s]])
                print(f'  {mde:>20.4f}', end='')
            else:
                print(f'  {"N/A":>20}', end='')
        print()

    # Save summary CSV
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('model,step,mde,chamfer,emd\n')
        for name in all_metrics:
            for s in sorted(all_metrics[name].keys()):
                vals = all_metrics[name][s]
                f.write(f'{name},{s},{np.mean([v["mde"] for v in vals]):.6f},'
                        f'{np.mean([v["chamfer"] for v in vals]):.6f},'
                        f'{np.mean([v["emd"] for v in vals]):.6f}\n')
    print(f'Saved: {csv_path}')

    print(f'\nAll results: {args.output_dir}')
    print(f'  eog {args.output_dir}/metric_curves.png &')
    print(f'  eog {args.output_dir}/ep*/step*.jpg &')


if __name__ == '__main__':
    main()
