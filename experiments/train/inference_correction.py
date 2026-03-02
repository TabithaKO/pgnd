#!/usr/bin/env python3
"""
inference_correction.py — Visual PGND + inference-time particle correction.

At each rollout step:
  1. PREDICT: Visual PGND forward pass (image-conditioned dynamics)
  2. CORRECT: Gradient descent on particle positions to minimize
     photometric loss against next-frame camera observation,
     regularized toward the dynamics prediction.
  3. Feed corrected state into next step.

Saves per-step metrics, debug panels, and a 5-column video:
  GT Image | GT Cloud | Pred Cloud (uncorrected) | Corrected Cloud | Overlay

Usage:
    cd ~/pgnd/experiments/train
    python inference_correction.py --episode 610
    python inference_correction.py --episode 620 --correction_steps 10 --correction_lr 0.001
    python inference_correction.py --episode 610 --lambda_reg 50.0 --no_visual
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
from tqdm import tqdm

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
from scipy.optimize import linear_sum_assignment

try:
    from neural_mesh_renderer import create_neural_mesh_renderer, project_vertex_colors
    from render_loss_ablation2 import PGNDCoordinateTransform
    from render_loss import GTImageLoader
    from build_cloth_mesh import compute_mesh_from_particles
    HAS_RENDERER = True
except ImportError as e:
    print(f"[FATAL] Neural renderer required: {e}")
    sys.exit(1)

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
    if pred.shape[0] > n: pred = pred[torch.randperm(pred.shape[0])[:n]]
    if gt.shape[0] > n: gt = gt[torch.randperm(gt.shape[0])[:n]]
    d = torch.norm(pred.unsqueeze(1) - gt.unsqueeze(0), dim=-1)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2

def compute_emd(pred, gt, n=1000):
    if pred.shape[0] > n: pred = pred[torch.randperm(pred.shape[0])[:n]]
    if gt.shape[0] > n: gt = gt[torch.randperm(gt.shape[0])[:n]]
    k = min(pred.shape[0], gt.shape[0])
    cost = np.linalg.norm(pred[:k].cpu().numpy()[:, None] - gt[:k].cpu().numpy()[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return cost[r, c].mean()

def find_latest_ckpt(ckpt_dir):
    ckpts = sorted(Path(ckpt_dir).glob('*.pt'))
    ckpts = [c for c in ckpts if c.stem.isdigit()]
    return ckpts[-1] if ckpts else None


# =============================================================================
# Episode context — camera, renderer, GT images, visual encoder
# =============================================================================

class InferenceContext:
    """All resources needed for inference + correction on one episode."""

    def __init__(self, cfg, episode, device, renderer_ckpt_path,
                 visual_ckpt_path=None, vis_dim=64, use_visual=True):
        self.cfg = cfg
        self.episode = episode
        self.device = device
        self.vis_dim = vis_dim

        log_root = root / 'log'
        source_root = log_root / str(cfg.train.source_dataset_name)
        ep_path = source_root / f'episode_{episode:04d}'

        # Calibration
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
        self.coord_transform = PGNDCoordinateTransform(cfg, ep_path).to_cuda()

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

        # GT image loader
        self.gt_loader = GTImageLoader(
            episode_dir=ep_dir, source_frame_start=frame_start,
            camera_id=1, image_size=(480, 848), skip_frame=load_skip * ds_skip)

        # Neural mesh renderer (for correction + viz)
        ckpt = torch.load(str(renderer_ckpt_path), map_location=device)
        self.renderer = create_neural_mesh_renderer(
            hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
            use_vertex_colors=True, use_view_direction=True, device=str(device))
        self.renderer.load_state_dict(ckpt['renderer'], strict=False)
        self.renderer.eval()
        # Keep renderer differentiable for correction (don't freeze)

        # Visual encoder
        self.vis_encoder = None
        if use_visual and HAS_VISUAL and visual_ckpt_path:
            try:
                self.vis_encoder = VisualEncoder(
                    model_name='dinov2_vits14', feature_dim=vis_dim,
                    camera_ids=[1], image_size=(480, 848), device=str(device))
                self.vis_encoder.setup_episode(
                    coord_transform=self.coord_transform,
                    cam_settings_dict={1: self.cam_settings})

                vis_ckpt = torch.load(str(visual_ckpt_path), map_location=device)
                if 'visual_proj' in vis_ckpt:
                    self.vis_encoder.proj.load_state_dict(vis_ckpt['visual_proj'])
                    print(f'  [vis] Loaded projection weights')
            except Exception as e:
                print(f'  [vis] Encoder setup failed: {e}')
                self.vis_encoder = None

    def get_gt_image(self, step):
        """Masked GT as (3, H, W) tensor on GPU, or None."""
        img = self.gt_loader.load_frame(step)
        if img is None:
            return None
        mask = self.gt_loader.load_mask(step)
        if mask is not None:
            img = img * mask
        return img

    def get_gt_rgb(self, step):
        """Masked GT as (H, W, 3) uint8 numpy RGB, or None."""
        img = self.get_gt_image(step)
        if img is None:
            return None
        return (img.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def render_and_loss(self, particles, gt_image):
        """
        Render particles, compute photometric loss against GT.
        Returns (loss_scalar, rendered_image_tensor).
        Particles must have requires_grad=True for correction.
        """
        mesh = compute_mesh_from_particles(particles.detach(), method='bpa')
        n_verts = mesh.pos.shape[0]
        verts = particles[:n_verts]  # keep grad connection
        vcols = project_vertex_colors(
            verts.detach(), gt_image, self.cam_settings, self.coord_transform)
        rendered, _ = self.renderer(
            verts, mesh.face, vcols, self.cam_settings, self.coord_transform)

        gt_mask = self.gt_loader.load_mask(0)  # reuse mask
        if gt_mask is not None:
            rendered = rendered * gt_mask
            gt_image = gt_image * gt_mask

        l1 = F.l1_loss(rendered, gt_image)
        try:
            from kornia.losses import ssim_loss
            ssim = ssim_loss(rendered.unsqueeze(0), gt_image.unsqueeze(0), window_size=11)
        except Exception:
            ssim = torch.tensor(0.0, device=particles.device)

        loss = l1 + 0.2 * ssim
        return loss, rendered

    def render_rgb(self, particles, step):
        """Render particles for visualization. Returns (H,W,3) uint8 RGB or None."""
        try:
            gt_img = self.gt_loader.load_frame(step)
            if gt_img is None:
                return None
            particles = particles.detach().to(self.device)
            mesh = compute_mesh_from_particles(particles, method='bpa')
            verts = particles[:mesh.pos.shape[0]]
            vcols = project_vertex_colors(verts, gt_img, self.cam_settings, self.coord_transform)
            rendered, _ = self.renderer(verts, mesh.face, vcols, self.cam_settings, self.coord_transform)
            return (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        except Exception:
            return None


# =============================================================================
# Correction step
# =============================================================================

def correct_particles(particles_pred, ctx, gt_image_next,
                      n_steps=5, lr=0.0005, lambda_reg=10.0):
    """
    Gradient descent on particle positions to minimize photometric loss.

    Args:
        particles_pred: (N, 3) predicted particles from dynamics (detached)
        ctx: InferenceContext with renderer + camera
        gt_image_next: (3, H, W) GT image at t+1 to correct toward
        n_steps: number of gradient steps
        lr: learning rate for particle correction
        lambda_reg: regularization toward dynamics prediction

    Returns:
        particles_corrected: (N, 3) corrected particles (detached)
        correction_log: dict with per-step losses
    """
    x_pred = particles_pred.detach().clone()
    x_corr = x_pred.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x_corr], lr=lr)

    log = {'render_loss': [], 'reg_loss': [], 'total_loss': [],
           'correction_norm': []}

    for k in range(n_steps):
        optimizer.zero_grad()

        try:
            render_loss, _ = ctx.render_and_loss(x_corr, gt_image_next)
        except Exception as e:
            if k == 0:
                print(f'    [correct] Render failed at step {k}: {e}')
            break

        # Regularization: don't drift far from dynamics prediction
        reg_loss = lambda_reg * (x_corr - x_pred).norm(dim=-1).mean()

        total = render_loss + reg_loss
        total.backward()
        optimizer.step()

        log['render_loss'].append(render_loss.item())
        log['reg_loss'].append(reg_loss.item())
        log['total_loss'].append(total.item())
        log['correction_norm'].append(
            (x_corr.detach() - x_pred).norm(dim=-1).mean().item())

    return x_corr.detach(), log


# =============================================================================
# Drawing
# =============================================================================

def draw_pc_cam(pts, h, w, color, cam, ct):
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
            cv2.circle(canvas, (uv[i], vv[i]), 2,
                       (int(c[2]*255), int(c[1]*255), int(c[0]*255)), -1)
    return canvas


def build_frame(step, num_steps, episode, mde_pred, mde_corr, correction_norm,
                gt_rgb, gt_pts, pred_pts, corr_pts,
                cam, ct, col_w=384, row_h=288):
    """
    6 columns:
    GT Image | GT Cloud | Pred Cloud (uncorrected) | Corrected Cloud | Overlay (GT+Corr) | Correction Δ
    """
    gap = 2
    header_h = 45
    n_cols = 6
    total_w = col_w * n_cols + gap * (n_cols - 1)
    total_h = header_h + row_h
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header
    line1 = f'Episode {episode} | Step {step}/{num_steps}'
    line2 = f'MDE pred={mde_pred:.4f}  corr={mde_corr:.4f}  delta={correction_norm:.5f}'
    cv2.putText(canvas, line1, (10, 18), font, 0.5, (255, 255, 255), 1)

    # Color the improvement
    if mde_corr < mde_pred:
        pct = (mde_pred - mde_corr) / mde_pred * 100
        color2 = (0, 220, 0)  # green = improved
        line2 += f'  ({pct:.1f}% better)'
    else:
        pct = (mde_corr - mde_pred) / mde_pred * 100
        color2 = (0, 0, 220)  # red = worse
        line2 += f'  ({pct:.1f}% worse)'
    cv2.putText(canvas, line2, (10, 36), font, 0.42, color2, 1)

    # Progress bar
    bar_x = total_w - 210; bar_w = 200; bar_h = 8; bar_y = 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * (step + 1) / max(num_steps, 1))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 100), -1)

    y = header_h

    def col_x(c):
        return c * (col_w + gap)

    # Col 0: GT Image
    if gt_rgb is not None:
        canvas[y:y+row_h, 0:col_w] = cv2.cvtColor(cv2.resize(gt_rgb, (col_w, row_h)), cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, 'GT Image', (5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 1: GT Cloud
    x1 = col_x(1)
    canvas[y:y+row_h, x1:x1+col_w] = draw_pc_cam(gt_pts, row_h, col_w, (0, 255, 128), cam, ct)
    cv2.putText(canvas, 'GT Cloud', (x1+5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 2: Predicted Cloud (uncorrected)
    x2 = col_x(2)
    canvas[y:y+row_h, x2:x2+col_w] = draw_pc_cam(pred_pts, row_h, col_w, (0, 100, 255), cam, ct)
    cv2.putText(canvas, f'Pred (MDE={mde_pred:.4f})', (x2+5, y+18), font, 0.35, (200, 200, 200), 1)

    # Col 3: Corrected Cloud
    x3 = col_x(3)
    canvas[y:y+row_h, x3:x3+col_w] = draw_pc_cam(corr_pts, row_h, col_w, (255, 150, 0), cam, ct)
    cv2.putText(canvas, f'Corrected (MDE={mde_corr:.4f})', (x3+5, y+18), font, 0.35, (200, 200, 200), 1)

    # Col 4: Overlay GT (cyan) + Corrected (orange)
    x4 = col_x(4)
    overlay = draw_pc_cam(gt_pts, row_h, col_w, (0, 255, 255), cam, ct)
    corr_layer = draw_pc_cam(corr_pts, row_h, col_w, (255, 150, 0), cam, ct)
    mask = corr_layer.any(axis=2)
    overlay[mask] = corr_layer[mask]
    canvas[y:y+row_h, x4:x4+col_w] = overlay
    cv2.putText(canvas, 'GT + Corrected', (x4+5, y+18), font, 0.4, (200, 200, 200), 1)

    # Col 5: Overlay GT (cyan) + Uncorrected (red) — for comparison
    x5 = col_x(5)
    overlay2 = draw_pc_cam(gt_pts, row_h, col_w, (0, 255, 255), cam, ct)
    pred_layer = draw_pc_cam(pred_pts, row_h, col_w, (0, 0, 255), cam, ct)
    mask2 = pred_layer.any(axis=2)
    overlay2[mask2] = pred_layer[mask2]
    canvas[y:y+row_h, x5:x5+col_w] = overlay2
    cv2.putText(canvas, 'GT + Uncorrected', (x5+5, y+18), font, 0.4, (200, 200, 200), 1)

    return canvas


# =============================================================================
# Main rollout with correction
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=610)
    parser.add_argument('--baseline_ckpt', default='cloth/train/ckpt/100000.pt')
    parser.add_argument('--visual_ckpt', default='latest')
    parser.add_argument('--renderer_ckpt', default='cloth/renderer_v4d/ckpt/200000.pt')
    parser.add_argument('--vis_dim', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=35)

    # Correction params
    parser.add_argument('--correction_steps', type=int, default=5,
                        help='Gradient steps per correction')
    parser.add_argument('--correction_lr', type=float, default=0.0005,
                        help='Learning rate for particle correction')
    parser.add_argument('--lambda_reg', type=float, default=10.0,
                        help='Regularization weight toward dynamics prediction')

    # Ablation flags
    parser.add_argument('--no_visual', action='store_true',
                        help='Use baseline PGNDModel instead of visual')
    parser.add_argument('--no_correction', action='store_true',
                        help='Skip correction (just visual PGND rollout for reference)')

    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    log_root = root / 'log'

    # Resolve visual ckpt
    use_visual = not args.no_visual
    if args.visual_ckpt == 'latest':
        ckpt_dir = log_root / 'cloth' / 'train_visual_v1' / 'ckpt'
        vis_path = find_latest_ckpt(ckpt_dir)
        if vis_path:
            args.visual_ckpt = str(vis_path)
            print(f'[visual] Latest: {vis_path.name}')
        else:
            print('[visual] No ckpt found, falling back to baseline')
            use_visual = False
            args.visual_ckpt = None

    # Determine dynamics checkpoint
    if use_visual and args.visual_ckpt:
        dyn_ckpt_path = args.visual_ckpt if os.path.isabs(args.visual_ckpt) \
            else str(log_root / args.visual_ckpt)
        vis_name = Path(dyn_ckpt_path).stem
        model_label = f'visual_{vis_name}'
    else:
        dyn_ckpt_path = str(log_root / args.baseline_ckpt)
        model_label = 'baseline_100k'
        use_visual = False

    renderer_path = str(log_root / args.renderer_ckpt)

    # Output
    corr_tag = f'K{args.correction_steps}_lr{args.correction_lr}_reg{args.lambda_reg}'
    if args.no_correction:
        corr_tag = 'no_correction'
    timestamp = time.strftime("%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = str(log_root / 'inference_correction' /
                              f'ep{args.episode:04d}_{model_label}_{corr_tag}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'frames'), exist_ok=True)
    print(f'Output: {args.output_dir}')

    # Config
    config_path = log_root / 'cloth' / 'train' / 'hydra.yaml'
    cfg = OmegaConf.create(yaml.load(open(config_path), Loader=yaml.CLoader))
    cfg.sim.num_steps = args.max_steps
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True

    # Init
    wp.init(); wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    device = torch.device('cuda:0')
    wp_dev = wp.get_device('cuda:0')

    # Setup context
    print(f'\nEpisode {args.episode}, model={model_label}, correction={not args.no_correction}')
    print(f'  correction_steps={args.correction_steps}, lr={args.correction_lr}, lambda_reg={args.lambda_reg}')

    ctx = InferenceContext(
        cfg, args.episode, device, renderer_path,
        visual_ckpt_path=dyn_ckpt_path if use_visual else None,
        vis_dim=args.vis_dim, use_visual=use_visual)

    # Load dynamics model
    ckpt = torch.load(dyn_ckpt_path, map_location=device)
    if use_visual and HAS_VISUAL:
        material = PGNDVisualModel(cfg, vis_dim=args.vis_dim)
        material.to(device)
        material.load_state_dict(ckpt['material'])
        print(f'  Loaded PGNDVisualModel from {Path(dyn_ckpt_path).name}')
    else:
        material = PGNDModel(cfg)
        material.to(device)
        material.load_state_dict(ckpt['material'])
        print(f'  Loaded PGNDModel from {Path(dyn_ckpt_path).name}')
    material.eval()

    friction = Friction(np.array([cfg.model.friction.value]))
    friction.to(device); friction.eval()

    # Load dataset
    source_root = log_root / str(cfg.train.source_dataset_name)
    dataset = RealTeleopBatchDataset(
        cfg, dataset_root=log_root / cfg.train.dataset_name / 'state',
        source_data_root=source_root, device=device,
        num_steps=cfg.sim.num_steps, eval_episode_name=f'episode_{args.episode:04d}')
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if cfg.sim.gripper_points:
        gp_ds = RealGripperDataset(cfg, device=device)
        gp_dl = DataLoader(gp_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

    sim = CacheDiffSimWithFrictionBatch(cfg, num_steps, bs, wp_dev, requires_grad=False)
    statics = StaticsBatch()
    statics.init(shape=(bs, num_particles), device=wp_dev)
    statics.update_clip_bound(clip_bound); statics.update_enabled(enabled)
    colliders = CollidersBatch()
    n_grip = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
    colliders.init(shape=(bs, n_grip), device=wp_dev)
    if n_grip > 0: colliders.initialize_grippers(actions[:, 0])
    enabled = enabled.to(device)

    # =========================================================================
    # Rollout with correction
    # =========================================================================
    print(f'\nRolling out {num_steps} steps...\n')

    metrics = {'step': [], 'mde_pred': [], 'mde_corr': [], 'chamfer_pred': [],
               'chamfer_corr': [], 'correction_norm': [], 'render_loss_before': [],
               'render_loss_after': []}
    frames = []

    for step in tqdm(range(num_steps), desc='Rollout+Correct'):
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

        # Visual features
        vis_feat = None
        if use_visual and ctx.vis_encoder is not None:
            try:
                img = ctx.gt_loader.load_frame(step)
                if img is not None:
                    x_cloth = x[:, :n_orig] if cfg.sim.gripper_points else x
                    vis_feat = ctx.vis_encoder(x_cloth, {1: img})
                    if cfg.sim.gripper_points:
                        pad = torch.zeros(vis_feat.shape[0], x.shape[1] - n_orig,
                                          args.vis_dim, device=device)
                        vis_feat = torch.cat([vis_feat, pad], dim=1)
            except Exception:
                vis_feat = None

        # PREDICT
        with torch.no_grad():
            if use_visual and HAS_VISUAL:
                pred = material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
            else:
                pred = material(x, v, x_his, v_his, enabled)
            x_new, v_new = sim(statics, colliders, step, x, v,
                               friction.mu[None].repeat(bs, 1), pred)

        # Extract cloth particles
        if cfg.sim.gripper_points:
            x_cloth_pred = x_new[0, :n_orig]
        else:
            x_cloth_pred = x_new[0]
        gt_pts = gt_x[0, step]

        mde_pred = compute_mde(x_cloth_pred, gt_pts)

        # CORRECT
        corr_norm = 0.0
        render_loss_before = 0.0
        render_loss_after = 0.0
        if not args.no_correction:
            gt_image_next = ctx.get_gt_image(step)
            if gt_image_next is not None:
                try:
                    # Render loss before correction
                    with torch.no_grad():
                        rl_before, _ = ctx.render_and_loss(
                            x_cloth_pred.clone().requires_grad_(False), gt_image_next)
                        render_loss_before = rl_before.item()
                except Exception:
                    render_loss_before = 0.0

                x_corrected, corr_log = correct_particles(
                    x_cloth_pred, ctx, gt_image_next,
                    n_steps=args.correction_steps,
                    lr=args.correction_lr,
                    lambda_reg=args.lambda_reg)

                corr_norm = (x_corrected - x_cloth_pred).norm(dim=-1).mean().item()
                if corr_log['render_loss']:
                    render_loss_after = corr_log['render_loss'][-1]

                # Write corrected particles back into state
                if cfg.sim.gripper_points:
                    x_new[0, :n_orig] = x_corrected
                else:
                    x_new[0] = x_corrected
            else:
                x_corrected = x_cloth_pred
        else:
            x_corrected = x_cloth_pred

        mde_corr = compute_mde(x_corrected, gt_pts)
        chamfer_pred = compute_chamfer(x_cloth_pred, gt_pts)
        chamfer_corr = compute_chamfer(x_corrected, gt_pts)

        # Log
        metrics['step'].append(step)
        metrics['mde_pred'].append(mde_pred)
        metrics['mde_corr'].append(mde_corr)
        metrics['chamfer_pred'].append(chamfer_pred)
        metrics['chamfer_corr'].append(chamfer_corr)
        metrics['correction_norm'].append(corr_norm)
        metrics['render_loss_before'].append(render_loss_before)
        metrics['render_loss_after'].append(render_loss_after)

        # Build frame
        gt_rgb = ctx.get_gt_rgb(step)
        frame = build_frame(
            step, num_steps, args.episode, mde_pred, mde_corr, corr_norm,
            gt_rgb, gt_pts.cpu(), x_cloth_pred.cpu(), x_corrected.cpu(),
            ctx.cam_settings, ctx.coord_transform)
        frames.append(frame)

        # Save frame image
        cv2.imwrite(os.path.join(args.output_dir, 'frames', f'step{step:03d}.jpg'), frame)

        # Print
        delta = mde_pred - mde_corr
        sign = '+' if delta > 0 else ''
        tqdm.write(f'  step {step:2d}: MDE pred={mde_pred:.4f} corr={mde_corr:.4f} '
                   f'({sign}{delta:.4f}) norm={corr_norm:.5f}')

        # Update state for next step
        x, v = x_new, v_new

        if cfg.sim.n_history > 0:
            if cfg.sim.gripper_points:
                xh = torch.cat([x_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:],
                                x[:, :n_orig, None]], dim=2)
                vh = torch.cat([v_his[:, :n_orig].reshape(bs, n_orig, -1, 3)[:, :, 1:],
                                v[:, :n_orig, None]], dim=2)
                x_his = xh.reshape(bs, n_orig, -1)
                v_his = vh.reshape(bs, n_orig, -1)
            else:
                x_his = torch.cat([x_his.reshape(bs, num_particles, -1, 3)[:, :, 1:],
                                   x[:, :, None]], dim=2).reshape(bs, num_particles, -1)
                v_his = torch.cat([v_his.reshape(bs, num_particles, -1, 3)[:, :, 1:],
                                   v[:, :, None]], dim=2).reshape(bs, num_particles, -1)

        if cfg.sim.gripper_points:
            x = x[:, :n_orig]; v = v[:, :n_orig]; enabled = enabled[:, :n_orig]

    # =========================================================================
    # Save video
    # =========================================================================
    video_path = os.path.join(args.output_dir, 'correction_rollout.mp4')
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f'\nVideo: {video_path} ({len(frames)} frames)')

    # =========================================================================
    # Save metrics + plots
    # =========================================================================

    # CSV
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('step,mde_pred,mde_corr,chamfer_pred,chamfer_corr,correction_norm,'
                'render_loss_before,render_loss_after\n')
        for i in range(len(metrics['step'])):
            f.write(f"{metrics['step'][i]},{metrics['mde_pred'][i]:.6f},"
                    f"{metrics['mde_corr'][i]:.6f},{metrics['chamfer_pred'][i]:.6f},"
                    f"{metrics['chamfer_corr'][i]:.6f},{metrics['correction_norm'][i]:.6f},"
                    f"{metrics['render_loss_before'][i]:.6f},{metrics['render_loss_after'][i]:.6f}\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MDE comparison
    ax = axes[0, 0]
    ax.plot(metrics['step'], metrics['mde_pred'], 'r-', linewidth=2, label='Predicted (uncorrected)')
    ax.plot(metrics['step'], metrics['mde_corr'], 'g-', linewidth=2, label='Corrected')
    ax.fill_between(metrics['step'], metrics['mde_corr'], metrics['mde_pred'],
                    alpha=0.2, color='green', where=[c < p for c, p in
                    zip(metrics['mde_corr'], metrics['mde_pred'])])
    ax.fill_between(metrics['step'], metrics['mde_corr'], metrics['mde_pred'],
                    alpha=0.2, color='red', where=[c >= p for c, p in
                    zip(metrics['mde_corr'], metrics['mde_pred'])])
    ax.set_xlabel('Step'); ax.set_ylabel('MDE')
    ax.set_title(f'MDE: Predicted vs Corrected (Episode {args.episode})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Chamfer
    ax = axes[0, 1]
    ax.plot(metrics['step'], metrics['chamfer_pred'], 'r-', linewidth=2, label='Predicted')
    ax.plot(metrics['step'], metrics['chamfer_corr'], 'g-', linewidth=2, label='Corrected')
    ax.set_xlabel('Step'); ax.set_ylabel('Chamfer')
    ax.set_title('Chamfer Distance'); ax.legend(); ax.grid(True, alpha=0.3)

    # Correction magnitude
    ax = axes[1, 0]
    ax.plot(metrics['step'], metrics['correction_norm'], 'b-', linewidth=2)
    ax.set_xlabel('Step'); ax.set_ylabel('Mean Correction Norm')
    ax.set_title('Correction Magnitude per Step'); ax.grid(True, alpha=0.3)

    # Render loss
    ax = axes[1, 1]
    ax.plot(metrics['step'], metrics['render_loss_before'], 'r--', linewidth=1.5, label='Before correction')
    ax.plot(metrics['step'], metrics['render_loss_after'], 'g-', linewidth=2, label='After correction')
    ax.set_xlabel('Step'); ax.set_ylabel('Render Loss')
    ax.set_title('Photometric Loss Before/After Correction'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Inference Correction | Episode {args.episode} | {model_label}\n'
                 f'K={args.correction_steps}, lr={args.correction_lr}, λ_reg={args.lambda_reg}',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'metrics.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Summary
    print(f'\n{"="*60}')
    print(f'Episode {args.episode} | {model_label} | K={args.correction_steps}')
    print(f'{"="*60}')
    avg_pred = np.mean(metrics['mde_pred'])
    avg_corr = np.mean(metrics['mde_corr'])
    pct = (avg_pred - avg_corr) / avg_pred * 100
    print(f'  Mean MDE predicted:  {avg_pred:.4f}')
    print(f'  Mean MDE corrected:  {avg_corr:.4f}')
    print(f'  Improvement:         {pct:+.1f}%')
    print(f'  Mean correction norm: {np.mean(metrics["correction_norm"]):.5f}')

    # Late-step improvement (steps 15+)
    late_mask = [s >= 15 for s in metrics['step']]
    if any(late_mask):
        late_pred = np.mean([m for m, ok in zip(metrics['mde_pred'], late_mask) if ok])
        late_corr = np.mean([m for m, ok in zip(metrics['mde_corr'], late_mask) if ok])
        late_pct = (late_pred - late_corr) / late_pred * 100
        print(f'  Late-step (≥15) MDE pred: {late_pred:.4f}  corr: {late_corr:.4f}  ({late_pct:+.1f}%)')

    print(f'\nResults: {args.output_dir}')
    print(f'  mpv {video_path}')
    print(f'  eog {plot_path}')


if __name__ == '__main__':
    main()
