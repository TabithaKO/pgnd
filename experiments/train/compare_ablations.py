"""
compare_ablations.py — Compare PGND ablation checkpoints
==========================================================

Evaluates multiple checkpoints on the same held-out episodes and produces:
1. 3D geometry metrics (MDE, Chamfer, EMD) per rollout step
2. Visual comparison panels:
   - GT point cloud vs predicted point cloud (overlaid, different colors)
   - GT image vs rendered image vs diff
   - Per-step metric curves across all models

Usage:
    cd ~/pgnd/experiments/train
    python compare_ablations.py

    # Or compare specific checkpoints:
    python compare_ablations.py --models baseline:cloth/train/ckpt/100000.pt ablation2:cloth/train_render_loss_mesh_gs_dino_v2/ckpt/055000.pt
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

# PGND imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import warp as wp
from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import get_root

from train_eval import transform_gripper_points, dataloader_wrapper
from torch.utils.data import DataLoader

try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Neural renderer imports for photorealistic rendering
try:
    from neural_mesh_renderer import (
        NeuralMeshRenderer, create_neural_mesh_renderer, project_vertex_colors,
    )
    from render_loss_ablation2 import PGNDCoordinateTransform
    from render_loss import setup_camera_for_render, GTImageLoader
    from build_cloth_mesh import compute_mesh_from_particles
    HAS_NEURAL_RENDERER = True
except ImportError as e:
    print(f"[compare_ablations] Neural renderer not available: {e}")
    HAS_NEURAL_RENDERER = False

# LBS-GS renderer not needed — baseline uses shared neural renderer
HAS_LBS_RENDERER = True

root: Path = get_root(__file__)


# =============================================================================
# Neural Mesh Rendering
# =============================================================================

class AblationRenderer:
    """Handles neural mesh rendering for the comparison script.

    Matches the rendering pipeline in render_loss_ablation2.py:
    particles → build mesh → project vertex colors → MLP forward → rasterize
    """

    def __init__(self, renderer_ckpt_path, cfg, episode, torch_device):
        self.device = torch_device
        self.cfg = cfg
        self.active = False

        if not HAS_NEURAL_RENDERER:
            print("  [AblationRenderer] Neural renderer not available")
            return

        log_root = root / 'log'

        # Load renderer weights from checkpoint
        ckpt = torch.load(str(log_root / renderer_ckpt_path), map_location=torch_device)
        if 'renderer' not in ckpt:
            print("  [AblationRenderer] No 'renderer' key in checkpoint")
            return

        self.renderer = create_neural_mesh_renderer(
            hidden_dim=256, n_hidden_layers=4, gaussians_per_face=8,
            use_vertex_colors=True, use_view_direction=True,
            device=str(torch_device),
        )
        # strict=False because checkpoint includes LPIPS VGG weights
        # which are not part of the renderer model itself
        self.renderer.load_state_dict(ckpt['renderer'], strict=False)
        self.renderer.eval()

        # Setup coordinate transform (per-episode)
        source_dataset_root = log_root / str(cfg.train.source_dataset_name)
        episode_data_path = source_dataset_root / f'episode_{episode:04d}'
        try:
            self.coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()
        except Exception as e:
            print(f"  [AblationRenderer] Coord transform failed: {e}")
            return

        # Resolve episode paths
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
            self.source_frame_start = int(meta[1]) + n_history * load_skip * ds_skip
            self.frame_skip = load_skip * ds_skip

            self.episode_dir = log_root / 'data_cloth' / recording_name / f'episode_{source_episode_id:04d}'
            self.camera_id = 1

            # Camera setup
            calib_dir = self.episode_dir / 'calibration'
            intr = np.load(str(calib_dir / 'intrinsics.npy'))
            rvec = np.load(str(calib_dir / 'rvecs.npy'))
            tvec = np.load(str(calib_dir / 'tvecs.npy'))

            R = cv2.Rodrigues(rvec[self.camera_id])[0]
            t = tvec[self.camera_id, :, 0]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            w2c = np.linalg.inv(c2w).astype(np.float32)

            self.cam_settings = {
                'w': 848, 'h': 480,
                'k': intr[self.camera_id],
                'w2c': w2c,
            }

            # Setup GT image loader (same as render_loss_ablation2)
            self.gt_loader = GTImageLoader(
                episode_dir=self.episode_dir,
                source_frame_start=self.source_frame_start,
                camera_id=self.camera_id,
                image_size=(480, 848),
                skip_frame=load_skip * ds_skip,
            )

        except Exception as e:
            print(f"  [AblationRenderer] Setup failed: {e}")
            return

        self.active = True
        print("  [AblationRenderer] Initialized successfully")

    @torch.no_grad()
    def render(self, particles, step):
        """Render particles into an RGB image.

        Follows the exact same pipeline as RenderLossModuleAblation2.compute_loss:
        1. Build mesh from particles (Delaunay)
        2. Project vertex colors from GT image
        3. Neural renderer MLP forward
        4. Return rendered image + GT image

        Args:
            particles: (N, 3) tensor in preprocessed space
            step: rollout step

        Returns:
            rendered_np: (H, W, 3) numpy uint8 RGB image, or None
            gt_np: (H, W, 3) numpy uint8 RGB image, or None
        """
        if not self.active:
            return None, None

        try:
            particles = particles.to(self.device)

            # Load GT image (for vertex color projection AND comparison)
            gt_image = self.gt_loader.load_frame(step)
            if gt_image is None:
                print(f"    [AblationRenderer] GT image not found for step {step} (rgb_dir={self.gt_loader.rgb_dir}, frame={self.gt_loader.frame_start + step * self.gt_loader.skip_frame})")
                return None, None

            gt_mask = self.gt_loader.load_mask(step)

            # Step 1: Build mesh from particles
            mesh_data = compute_mesh_from_particles(particles, method='bpa')
            faces = mesh_data.face  # (3, N_faces)
            n_verts = mesh_data.pos.shape[0]
            vertices = particles[:n_verts]

            # Step 2: Project vertex colors from GT image
            vertex_colors = project_vertex_colors(
                vertices_preproc=vertices,
                image=gt_image,
                cam_settings=self.cam_settings,
                coord_transform=self.coord_transform,
            )

            # Step 3: Neural renderer forward
            try:
                rendered_image, debug_info = self.renderer(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors,
                    cam_settings=self.cam_settings,
                    coord_transform=self.coord_transform,
                )
            except Exception as render_err:
                print(f"    [AblationRenderer] MLP forward failed: {render_err}")
                import traceback
                traceback.print_exc()
                return None, None

            # Convert to numpy
            rendered_np = rendered_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            rendered_np = (rendered_np * 255).astype(np.uint8)

            # GT image: apply mask for fair comparison
            gt_np = gt_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            if gt_mask is not None:
                mask_np = gt_mask.detach().cpu().permute(1, 2, 0).numpy()
                gt_np = gt_np * mask_np
            gt_np = (gt_np * 255).astype(np.uint8)

            return rendered_np, gt_np

        except Exception as e:
            print(f"    [AblationRenderer] Render failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            return None, None


class BaselineLBSRenderer:
    """Renders baseline predictions using the LBS-GS pipeline from render_loss.py.

    Loads frozen per-frame Gaussian Splats, deforms them via LBS based on
    particle displacements, and renders. Uses INCREMENTAL deformation
    (step-by-step) exactly like training, so splats stay photorealistic.
    """

    def __init__(self, cfg, episode, torch_device):
        self.device = torch_device
        self.cfg = cfg
        self.active = False

        log_root = root / 'log'

        try:
            from render_loss import (
                read_splat_raw, setup_camera_for_render, DifferentiableLBS,
                GTImageLoader,
            )
            from render_loss_ablation2 import PGNDCoordinateTransform
        except ImportError as e:
            print(f"  [BaselineLBSRenderer] Import failed: {e}")
            return

        try:
            # Resolve episode paths
            source_dataset_root = log_root / str(cfg.train.source_dataset_name)
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
            gs_dir = episode_dir / 'gs'

            if not gs_dir.exists():
                print(f"  [BaselineLBSRenderer] No GS dir: {gs_dir}")
                return

            splat_files = sorted(gs_dir.glob('*.splat'))
            if not splat_files:
                print(f"  [BaselineLBSRenderer] No splat files")
                return

            frame_nums = [int(f.stem) for f in splat_files]
            closest_frame = min(frame_nums, key=lambda x: abs(x - source_frame_start))
            gs_path = gs_dir / f'{closest_frame:06d}.splat'

            splat_data = read_splat_raw(str(gs_path))
            opa = splat_data['opacities'][:, 0]
            valid = opa > 0.1

            self.gs_means3D_orig = torch.from_numpy(splat_data['pts'][valid]).to(torch_device)
            self.gs_colors = torch.from_numpy(splat_data['colors'][valid]).to(torch_device)
            self.gs_scales = torch.from_numpy(splat_data['scales'][valid]).to(torch_device)
            self.gs_quats = torch.from_numpy(splat_data['quats'][valid]).to(torch_device)
            self.gs_opacities = torch.from_numpy(splat_data['opacities'][valid]).to(torch_device)

            # Transform GS to preprocessed space
            self.coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()
            R_cpu = self.coord_transform.R.cpu()
            t_cpu = self.coord_transform.translation.cpu()
            gs_pts = self.gs_means3D_orig.cpu()
            gs_pts = gs_pts @ R_cpu
            gs_pts = gs_pts * self.coord_transform.scale
            gs_pts = gs_pts + t_cpu
            self.gs_means3D_preproc = gs_pts.to(torch_device)

            camera_id = 1
            self.camera_id = camera_id
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

            self.cam_settings = {
                'w': 848, 'h': 480,
                'k': intr[camera_id],
                'w2c': w2c,
            }

            self.gt_loader = GTImageLoader(
                episode_dir=episode_dir,
                source_frame_start=source_frame_start,
                camera_id=camera_id,
                image_size=(480, 848),
                skip_frame=load_skip * ds_skip,
            )

            # Store preprocessing params for inverse transform
            # IMPORTANT: The GS forward transform is pts @ R (not R.T like particles)
            # So the inverse is pts @ R.T (not pts @ R like PGNDCoordinateTransform)
            self._preproc_R = self.coord_transform.R.clone()  # on CUDA
            self._preproc_scale = self.coord_transform.scale
            self._preproc_translation = self.coord_transform.translation.clone()  # on CUDA

            self.lbs = DifferentiableLBS(k_neighbors=16)
            self._rendered_cache = {}
            self._gt_cache = {}

            self.active = True
            print(f"  [BaselineLBSRenderer] Initialized ({self.gs_means3D_orig.shape[0]} gaussians)")

        except Exception as e:
            print(f"  [BaselineLBSRenderer] Init failed: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def precompute_all_steps(self, all_pred_positions, all_gt_positions, viz_steps):
        """Run incremental LBS through ALL rollout steps, cache renders at viz_steps.

        Uses GT initial positions for LBS initialization (same as training),
        then deforms incrementally using predicted positions.
        """
        if not self.active:
            return

        try:
            from render_loss import setup_camera_for_render
            from diff_gaussian_rasterization import GaussianRasterizer

            viz_set = set(viz_steps)

            # Use GT initial positions for LBS init (same as training)
            gt_particles_0 = all_gt_positions[0].to(self.device)
            self.lbs.precompute(gt_particles_0, self.gs_means3D_preproc)

            # Start from GT initial state (same as training)
            particles_prev = gt_particles_0.clone()
            gs_xyz_curr = self.gs_means3D_preproc.clone()
            gs_quat_curr = self.gs_quats.clone()

            for step in range(len(all_pred_positions)):
                particles_curr = all_pred_positions[step].to(self.device)

                gs_xyz_new, gs_quat_new = self.lbs.deform(
                    bones_prev=particles_prev,
                    bones_curr=particles_curr,
                    gs_xyz_prev=gs_xyz_curr,
                    gs_quat_prev=gs_quat_curr,
                )

                if step in viz_set:
                    # Inverse transform: preprocessed → world
                    # Training does: gs_xyz_render = (gs_xyz_new - t) / s @ R.T
                    # NOT PGNDCoordinateTransform.inverse_transform which uses @ R
                    gs_xyz_render = (gs_xyz_new - self._preproc_translation) / self._preproc_scale
                    gs_xyz_render = gs_xyz_render @ self._preproc_R.T

                    render_data = {
                        'means3D': gs_xyz_render,
                        'colors_precomp': self.gs_colors,
                        'rotations': torch.nn.functional.normalize(gs_quat_new, dim=-1),
                        'opacities': self.gs_opacities,
                        'scales': self.gs_scales,
                        'means2D': torch.zeros_like(gs_xyz_render) + 0,
                    }

                    cam = setup_camera_for_render(
                        w=self.cam_settings['w'], h=self.cam_settings['h'],
                        k=self.cam_settings['k'], w2c=self.cam_settings['w2c'],
                    )

                    rendered, _, _ = GaussianRasterizer(raster_settings=cam)(**render_data)

                    rendered_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    self._rendered_cache[step] = (rendered_np * 255).astype(np.uint8)

                    gt_image = self.gt_loader.load_frame(step)
                    gt_mask = self.gt_loader.load_mask(step)
                    if gt_image is not None:
                        gt_np = gt_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        if gt_mask is not None:
                            mask_np = gt_mask.detach().cpu().permute(1, 2, 0).numpy()
                            gt_np = gt_np * mask_np
                        self._gt_cache[step] = (gt_np * 255).astype(np.uint8)

                particles_prev = particles_curr.clone()
                gs_xyz_curr = gs_xyz_new.clone()
                gs_quat_curr = gs_quat_new.clone()

            print(f"  [BaselineLBSRenderer] Precomputed {len(self._rendered_cache)} renders")

        except Exception as e:
            print(f"  [BaselineLBSRenderer] Precompute failed: {e}")
            import traceback
            traceback.print_exc()

    def render(self, particles, step):
        """Return cached render for this step."""
        if not self.active:
            return None, None
        rendered_np = self._rendered_cache.get(step)
        gt_np = self._gt_cache.get(step)
        return rendered_np, gt_np


# =============================================================================
# Metrics
# =============================================================================

def compute_mde(pred, gt):
    """Mean Displacement Error: mean L2 distance between corresponding particles."""
    return torch.norm(pred - gt, dim=-1).mean().item()


def compute_chamfer(pred, gt, subsample=2000):
    """Chamfer distance between two point sets."""
    # Subsample for speed
    if pred.shape[0] > subsample:
        idx = torch.randperm(pred.shape[0])[:subsample]
        pred = pred[idx]
    if gt.shape[0] > subsample:
        idx = torch.randperm(gt.shape[0])[:subsample]
        gt = gt[idx]

    # pred -> gt
    diff_p2g = pred.unsqueeze(1) - gt.unsqueeze(0)  # (N, M, 3)
    dist_p2g = torch.norm(diff_p2g, dim=-1)  # (N, M)
    chamfer_p2g = dist_p2g.min(dim=1).values.mean()

    # gt -> pred
    chamfer_g2p = dist_p2g.min(dim=0).values.mean()

    return (chamfer_p2g + chamfer_g2p).item() / 2.0


def compute_emd_approx(pred, gt, subsample=1000):
    """Approximate Earth Mover's Distance via linear assignment on subsampled sets."""
    if not HAS_SCIPY:
        return 0.0

    if pred.shape[0] > subsample:
        idx = torch.randperm(pred.shape[0])[:subsample]
        pred = pred[idx]
    if gt.shape[0] > subsample:
        idx = torch.randperm(gt.shape[0])[:subsample]
        gt = gt[idx]

    # Match sizes
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n].cpu().numpy()
    gt = gt[:n].cpu().numpy()

    cost = np.linalg.norm(pred[:, None] - gt[None, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].mean()


# =============================================================================
# Rollout a single episode with a given checkpoint
# =============================================================================

def rollout_episode(cfg, ckpt_path, episode, torch_device, wp_device):
    """Run a single episode rollout and return predicted + GT positions per step.

    Returns:
        pred_positions: list of (N, 3) tensors, one per step
        gt_positions: list of (N, 3) tensors, one per step
    """
    log_root = root / 'log'

    # Load dataset for this episode
    source_dataset_root = log_root / str(cfg.train.source_dataset_name)
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

    # Load checkpoint
    ckpt = torch.load(str(log_root / ckpt_path), map_location=torch_device)
    material = PGNDModel(cfg)
    material.to(torch_device)
    material.load_state_dict(ckpt['material'])
    material.requires_grad_(False)
    material.eval()

    if 'friction' in ckpt:
        friction = ckpt['friction']['mu'].reshape(-1, 1)
    else:
        friction = torch.tensor(cfg.model.friction.value, device=torch_device).reshape(-1, 1)

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
    gt_v = gt_v.to(torch_device)

    batch_size = gt_x.shape[0]
    num_steps_total = gt_x.shape[1]
    num_particles = gt_x.shape[2]

    if cfg.sim.gripper_points:
        num_gripper_particles = gripper_x.shape[2]
        num_particles_orig = num_particles
        num_particles = num_particles + num_gripper_particles

    cfg_copy = cfg.copy()
    cfg_copy.sim.num_steps = num_steps_total
    sim = CacheDiffSimWithFrictionBatch(cfg_copy, num_steps_total, batch_size, wp_device, requires_grad=False)

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
    enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

    pred_positions = []
    gt_positions = []

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
                x = x[:, :num_particles_orig]
                v = v[:, :num_particles_orig]
                enabled = enabled[:, :num_particles_orig]

            pred_positions.append(x[0].cpu())
            gt_positions.append(gt_x[0, step].cpu())

    return pred_positions, gt_positions


# =============================================================================
# Visualization
# =============================================================================

def project_points_to_image(points_3d, K, w2c, h, w):
    """Project 3D points to 2D pixel coordinates."""
    R_cam = w2c[:3, :3]
    t_cam = w2c[:3, 3]
    pts_cam = (R_cam @ points_3d.T + t_cam.reshape(3, 1)).T
    pts_2d = (K @ pts_cam.T).T
    u = (pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)).astype(int)
    v = (pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)).astype(int)
    valid = (pts_cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def draw_point_cloud(pts, h=480, w=424, color=(255, 255, 0), label=None,
                     cam_settings=None, coord_transform=None):
    """Draw a single point cloud on black background.
    
    If cam_settings and coord_transform are provided, projects to camera view
    (matching training debug images). Otherwise falls back to XZ projection.
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts_np = pts.numpy() if torch.is_tensor(pts) else pts

    if cam_settings is not None and coord_transform is not None:
        # Project to camera view (same as training debug images)
        pts_t = torch.tensor(pts_np, dtype=torch.float32).cuda()
        pts_world = coord_transform.inverse_transform(pts_t).cpu().numpy()

        K = cam_settings['k']
        w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]

        pts_cam = (R_cam @ pts_world.T + t_cam.reshape(3, 1)).T
        pts_2d = (K @ pts_cam.T).T
        u_full = pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)
        v_full = pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)
        depth = pts_cam[:, 2]
        valid = (depth > 0) & (u_full >= 0) & (u_full < 848) & (v_full >= 0) & (v_full < 480)

        # Scale to canvas size
        u = (u_full / 848.0 * w).astype(int)
        v = (v_full / 480.0 * h).astype(int)

        # Depth coloring
        if valid.sum() > 0:
            d_valid = depth[valid]
            d_min, d_max = d_valid.min(), d_valid.max()
            d_norm = (d_valid - d_min) / (d_max - d_min + 1e-8)

            u_v, v_v = u[valid], v[valid]
            for i in range(len(u_v)):
                if 0 <= u_v[i] < w and 0 <= v_v[i] < h:
                    # Blend base color with depth
                    base = np.array(color, dtype=float) / 255.0
                    c = base * (1.0 - 0.3 * d_norm[i])
                    cv2.circle(canvas, (u_v[i], v_v[i]), 2,
                               (int(c[0]*255), int(c[1]*255), int(c[2]*255)), -1)
    else:
        # Fallback: XZ projection
        mins = pts_np.min(axis=0)
        maxs = pts_np.max(axis=0)
        span = (maxs - mins).max() + 1e-8
        margin = 15

        u = ((pts_np[:, 0] - mins[0]) / span * (w - 2 * margin) + margin).astype(int)
        v = ((pts_np[:, 2] - mins[2]) / span * (h - 2 * margin - 15) + margin + 15).astype(int)

        for i in range(len(u)):
            if 0 <= u[i] < w and 0 <= v[i] < h:
                cv2.circle(canvas, (u[i], v[i]), 2, color, -1)

    if label:
        cv2.putText(canvas, label, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return canvas


def draw_point_cloud_overlay(pred_pts, gt_pts, h=480, w=424,
                              cam_settings=None, coord_transform=None):
    """Draw GT (cyan) and predicted (red) point clouds overlaid."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    pred_np = pred_pts.numpy() if torch.is_tensor(pred_pts) else pred_pts
    gt_np = gt_pts.numpy() if torch.is_tensor(gt_pts) else gt_pts

    if cam_settings is not None and coord_transform is not None:
        # Camera projection for both
        K = cam_settings['k']
        w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]

        def project(pts_np):
            pts_t = torch.tensor(pts_np, dtype=torch.float32).cuda()
            pts_world = coord_transform.inverse_transform(pts_t).cpu().numpy()
            pts_cam = (R_cam @ pts_world.T + t_cam.reshape(3, 1)).T
            pts_2d = (K @ pts_cam.T).T
            u = (pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8) / 848.0 * w).astype(int)
            v = (pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8) / 480.0 * h).astype(int)
            valid = (pts_cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            return u, v, valid

        gt_u, gt_v, gt_valid = project(gt_np)
        pred_u, pred_v, pred_valid = project(pred_np)

        for i in range(len(gt_u)):
            if gt_valid[i]:
                cv2.circle(canvas, (gt_u[i], gt_v[i]), 2, (255, 255, 0), -1)
        for i in range(len(pred_u)):
            if pred_valid[i]:
                cv2.circle(canvas, (pred_u[i], pred_v[i]), 2, (0, 0, 255), -1)
    else:
        # Fallback XZ
        all_pts = np.concatenate([pred_np, gt_np], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = (maxs - mins).max() + 1e-8
        margin = 15

        def to_px(pts):
            u = ((pts[:, 0] - mins[0]) / span * (w - 2 * margin) + margin).astype(int)
            v = ((pts[:, 2] - mins[2]) / span * (h - 2 * margin - 15) + margin + 15).astype(int)
            return u, v

        gt_u, gt_v = to_px(gt_np)
        for i in range(len(gt_u)):
            if 0 <= gt_u[i] < w and 0 <= gt_v[i] < h:
                cv2.circle(canvas, (gt_u[i], gt_v[i]), 2, (255, 255, 0), -1)

        pred_u, pred_v = to_px(pred_np)
        for i in range(len(pred_u)):
            if 0 <= pred_u[i] < w and 0 <= pred_v[i] < h:
                cv2.circle(canvas, (pred_u[i], pred_v[i]), 2, (0, 0, 255), -1)

    cv2.putText(canvas, 'Overlay', (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, 'GT', (w - 70, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    cv2.putText(canvas, 'Pred', (w - 40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return canvas


def load_gt_image(cfg, episode, step, camera_id=1):
    """Load GT image using GTImageLoader (same paths as training)."""
    try:
        from render_loss import GTImageLoader
        log_root = get_root(__file__) / 'log'
        source_dataset_root = log_root / str(cfg.train.source_dataset_name)
        meta = np.loadtxt(str(source_dataset_root / f'episode_{episode:04d}' / 'meta.txt'))

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

        gt_loader = GTImageLoader(
            episode_dir=episode_dir,
            source_frame_start=source_frame_start,
            camera_id=camera_id,
            image_size=(480, 848),
            skip_frame=load_skip * ds_skip,
        )

        gt_image = gt_loader.load_frame(step)
        gt_mask = gt_loader.load_mask(step)

        if gt_image is None:
            return None

        gt_np = gt_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        if gt_mask is not None:
            mask_np = gt_mask.detach().cpu().permute(1, 2, 0).numpy()
            gt_np = gt_np * mask_np
        gt_np = (gt_np * 255).astype(np.uint8)
        return gt_np

    except Exception as e:
        print(f"  [load_gt_image] Failed: {e}")
        return None


def make_comparison_panel(model_results, step, episode, cfg=None,
                          renderers=None, K=None, w2c=None, h=480, w=848):
    """Create comparison panel with point clouds AND rendered images.

    Layout:
      Row 0 (observation):  [GT Image @ t  | GT PC @ t  | GT Image @ t+1 | GT PC @ t+1]
      Per model:            [Pred PC       | Overlay     | Rendered        | Diff vs GT]
    
    Shows the observation at time t (input) and t+1 (target), plus each
    model's prediction at t+1.
    """
    if renderers is None:
        renderers = {}
    n_models = len(model_results)
    col_w = w // 4
    row_h = min(h, 300)

    total_h = 25 + row_h + n_models * (row_h + 25) + 10
    total_w = col_w * 4 + 15

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    cv2.putText(canvas, f'Episode {episode}, Step {step}', (10, 18), font, 0.6, (255, 255, 255), 1)
    y += 25

    # Get camera params from renderers for point cloud projection
    _cam_settings = None
    _coord_transform = None
    if renderers:
        for r in renderers.values():
            if r is not None and r.active and hasattr(r, 'cam_settings') and hasattr(r, 'coord_transform'):
                _cam_settings = r.cam_settings
                _coord_transform = r.coord_transform
                break

    first_model_data = list(model_results.values())[0]
    gt_pts_t1 = first_model_data['gt_positions'][step]  # step N (target)

    # t = step N-1 (what model sees), t+1 = step N (what model predicts)
    prev_step = max(0, step - 1)
    gt_pts_t0 = first_model_data['gt_positions'][prev_step]

    # Debug: verify positions are different
    if step > 0:
        pos_diff = torch.norm(gt_pts_t1 - gt_pts_t0, dim=-1).mean().item()
        print(f"    [Panel] GT PC diff between step {prev_step} and {step}: {pos_diff:.6f}")

    # Load GT images at step N-1 and step N
    gt_img_t0 = load_gt_image(cfg, episode, prev_step) if cfg is not None else None
    gt_img_t1 = load_gt_image(cfg, episode, step) if cfg is not None else None

    # Debug: verify images are different
    if gt_img_t0 is not None and gt_img_t1 is not None and step > 0:
        img_diff = np.abs(gt_img_t0.astype(float) - gt_img_t1.astype(float)).mean()
        print(f"    [Panel] GT image diff between step {prev_step} and {step}: {img_diff:.2f}")

    # Header row: [GT Image t=N-1 | GT PC t=N-1 | GT Image t=N | GT PC t=N]
    # Col 0: GT Image at previous step
    if gt_img_t0 is not None:
        img_resized = cv2.resize(gt_img_t0, (col_w, row_h))
        canvas[y:y + row_h, 0:col_w] = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, f'Obs t={prev_step}', (5, y + 15), font, 0.35, (200, 200, 200), 1)
    else:
        placeholder = np.full((row_h, col_w, 3), 20, dtype=np.uint8)
        cv2.putText(placeholder, 'No image', (5, row_h // 2), font, 0.4, (100, 100, 100), 1)
        canvas[y:y + row_h, 0:col_w] = placeholder

    # Col 1: GT Point Cloud at step N-1
    gt_pc_t0 = draw_point_cloud(gt_pts_t0, h=row_h, w=col_w, color=(0, 255, 128),
                                 label=f'GT PC t={prev_step}',
                                 cam_settings=_cam_settings, coord_transform=_coord_transform)
    canvas[y:y + row_h, col_w + 5:col_w * 2 + 5] = gt_pc_t0

    # Col 2: GT Image at time t+1
    if gt_img_t1 is not None:
        img_resized = cv2.resize(gt_img_t1, (col_w, row_h))
        canvas[y:y + row_h, col_w * 2 + 10:col_w * 3 + 10] = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, f'GT t={step}', (col_w * 2 + 15, y + 15), font, 0.35, (200, 200, 200), 1)
    else:
        placeholder = np.full((row_h, col_w, 3), 20, dtype=np.uint8)
        cv2.putText(placeholder, 'No image', (5, row_h // 2), font, 0.4, (100, 100, 100), 1)
        canvas[y:y + row_h, col_w * 2 + 10:col_w * 3 + 10] = placeholder

    # Col 3: GT Point Cloud at time t+1
    gt_pc_t1 = draw_point_cloud(gt_pts_t1, h=row_h, w=col_w, color=(255, 255, 0),
                                 label=f'GT PC t={step}',
                                 cam_settings=_cam_settings, coord_transform=_coord_transform)
    canvas[y:y + row_h, col_w * 3 + 15:col_w * 4 + 15] = gt_pc_t1

    y += row_h + 5

    # Per model rows
    for model_name, data in model_results.items():
        pred_pts = data['pred_positions'][step]
        gt_pts_model = data['gt_positions'][step]

        mde = compute_mde(pred_pts, gt_pts_model)
        chamfer = compute_chamfer(pred_pts, gt_pts_model)
        emd = compute_emd_approx(pred_pts, gt_pts_model)

        cv2.putText(canvas, model_name, (5, y + 13), font, 0.5, (0, 255, 255), 1)

        # Quality indicator next to name
        if mde < 0.1:
            cv2.putText(canvas, 'GOOD', (len(model_name) * 11 + 15, y + 13), font, 0.4, (0, 255, 0), 1)
        elif mde < 0.3:
            cv2.putText(canvas, 'OK', (len(model_name) * 11 + 15, y + 13), font, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(canvas, 'POOR', (len(model_name) * 11 + 15, y + 13), font, 0.4, (0, 0, 255), 1)

        # Metrics next to name
        metrics_text = f'MDE:{mde:.4f}  CD:{chamfer:.4f}  EMD:{emd:.4f}'
        cv2.putText(canvas, metrics_text, (total_w // 2, y + 13), font, 0.35, (180, 180, 180), 1)
        y += 18

        # Col 1: Predicted point cloud
        pred_pc = draw_point_cloud(pred_pts, h=row_h, w=col_w, color=(0, 0, 255), label='Predicted',
                                    cam_settings=_cam_settings, coord_transform=_coord_transform)
        canvas[y:y + row_h, 0:col_w] = pred_pc

        # Col 2: Overlay
        overlay = draw_point_cloud_overlay(pred_pts, gt_pts_model, h=row_h, w=col_w,
                                          cam_settings=_cam_settings, coord_transform=_coord_transform)
        canvas[y:y + row_h, col_w + 5:col_w * 2 + 5] = overlay

        # Col 3 & 4: Rendered image + Diff (if renderer available for this model)
        rendered_np, gt_render_np = None, None
        if renderers:
            if 'neural' in model_name.lower() or 'v4b' in model_name.lower():
                r = renderers.get('ablation')
            elif 'baseline' in model_name.lower():
                r = renderers.get('baseline')
            else:
                r = None
            if r is not None:
                rendered_np, gt_render_np = r.render(pred_pts, step)
                if rendered_np is None:
                    print(f"    [{model_name}] Render returned None at step {step}")

        if rendered_np is not None:
            # Show rendered image
            rendered_resized = cv2.resize(rendered_np, (col_w, row_h))
            rendered_bgr = cv2.cvtColor(rendered_resized, cv2.COLOR_RGB2BGR)
            canvas[y:y + row_h, col_w * 2 + 10:col_w * 3 + 10] = rendered_bgr
            cv2.putText(canvas, 'Rendered', (col_w * 2 + 15, y + 15), font, 0.35, (200, 200, 200), 1)

            # Show diff vs GT
            if gt_render_np is not None:
                gt_resized_diff = cv2.resize(gt_render_np, (col_w, row_h))
                rendered_resized_rgb = cv2.resize(rendered_np, (col_w, row_h))
                diff = np.abs(rendered_resized_rgb.astype(float) - gt_resized_diff.astype(float))
                diff_amplified = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
                diff_bgr = cv2.cvtColor(diff_amplified, cv2.COLOR_RGB2BGR)
                canvas[y:y + row_h, col_w * 3 + 15:col_w * 4 + 15] = diff_bgr
                cv2.putText(canvas, 'Diff (3x)', (col_w * 3 + 20, y + 15), font, 0.35, (200, 200, 200), 1)
        else:
            # No renderer — show GT point cloud again for reference
            gt_pc = draw_point_cloud(gt_pts_model, h=row_h, w=col_w, color=(255, 255, 0), label='GT',
                                    cam_settings=_cam_settings, coord_transform=_coord_transform)
            canvas[y:y + row_h, col_w * 2 + 10:col_w * 3 + 10] = gt_pc

            # Metrics panel
            metrics_img = np.zeros((row_h, col_w, 3), dtype=np.uint8)
            cv2.putText(metrics_img, f'MDE:     {mde:.6f}', (10, 40), font, 0.5, (255, 255, 255), 1)
            cv2.putText(metrics_img, f'Chamfer: {chamfer:.6f}', (10, 70), font, 0.5, (255, 255, 255), 1)
            cv2.putText(metrics_img, f'EMD:     {emd:.6f}', (10, 100), font, 0.5, (255, 255, 255), 1)
            canvas[y:y + row_h, col_w * 3 + 15:col_w * 4 + 15] = metrics_img

        y += row_h + 5

    return canvas


def plot_metric_curves(all_metrics, save_path):
    """Plot MDE, Chamfer, EMD curves for all models over rollout steps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_names = ['MDE', 'Chamfer', 'EMD']

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

    for ax, metric_name in zip(axes, metric_names):
        for i, (model_name, metrics) in enumerate(all_metrics.items()):
            key = metric_name.lower()
            values = [m[key] for m in metrics]
            steps = range(len(values))
            color = colors[i % len(colors)]
            ax.plot(steps, values, label=model_name, color=color, alpha=0.8)
        ax.set_xlabel('Rollout Step')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f'  Saved metric curves: {save_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare PGND ablation checkpoints')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model specifications as name:ckpt_path pairs')
    parser.add_argument('--episodes', nargs='+', type=int, default=[610, 615, 620, 625, 630],
                        help='Episode indices to evaluate')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Max rollout steps to evaluate')
    parser.add_argument('--viz_steps', nargs='+', type=int, default=[0, 10, 20, 30, 40],
                        help='Steps at which to save visual comparisons')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    # Default model configurations
    if args.models is None:
        args.models = [
            'baseline:cloth/train/ckpt/100000.pt',
            'ablation1_dino:cloth/train_render_loss_dino/ckpt/100000.pt',
            'ablation2_mesh_gs:cloth/train_render_loss_mesh_gs_dino/ckpt/100000.pt',
            'ablation2_neural_v4b:cloth/train_render_loss_mesh_gs_dino_v2/ckpt/055000.pt',
        ]

    # Parse model specs
    models = {}
    for spec in args.models:
        name, ckpt = spec.split(':')
        models[name] = ckpt

    # Load config from baseline
    log_root = root / 'log'
    baseline_ckpt = list(models.values())[0]
    train_name = '/'.join(baseline_ckpt.split('/')[:-2])  # e.g., cloth/train
    config_path = log_root / train_name / 'hydra.yaml'
    if not config_path.exists():
        # Try finding any config
        for ckpt in models.values():
            tn = '/'.join(ckpt.split('/')[:-2])
            cp = log_root / tn / 'hydra.yaml'
            if cp.exists():
                config_path = cp
                break

    print(f'Loading config from: {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    cfg = OmegaConf.create(config)
    cfg.sim.num_steps = args.max_steps
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True

    # Output directory
    if args.output_dir is None:
        args.output_dir = str(log_root / 'ablation_comparison' / time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Init warp
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    gpus = [int(gpu) for gpu in cfg.gpus]
    wp_device = wp.get_device(f'cuda:{gpus[0]}')
    torch_device = torch.device(f'cuda:{gpus[0]}')

    # Try to load camera parameters for projection
    K, w2c = None, None
    try:
        source_dataset_root = log_root / str(cfg.train.source_dataset_name)
        meta = np.loadtxt(str(source_dataset_root / f'episode_{args.episodes[0]:04d}' / 'meta.txt'))
        with open(source_dataset_root / 'metadata.json') as f:
            metadata = json.load(f)
        entry = metadata[args.episodes[0]]
        source_data_dir = Path(entry['path'])
        recording_name = source_data_dir.parent.name
        source_episode_id = int(meta[0])
        calib_dir = log_root / 'data_cloth' / recording_name / f'episode_{source_episode_id:04d}' / 'calibration'
        if calib_dir.exists():
            K = np.load(str(calib_dir / 'intrinsics.npy'))[1]
            rvec = np.load(str(calib_dir / 'rvecs.npy'))
            tvec = np.load(str(calib_dir / 'tvecs.npy'))
            R = cv2.Rodrigues(rvec[1])[0]
            t = tvec[1, :, 0]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            w2c = np.linalg.inv(c2w).astype(np.float32)
            print(f'  Camera params loaded from {calib_dir}')
    except Exception as e:
        print(f'  Could not load camera params: {e}')

    # Verify checkpoints exist
    for name, ckpt in models.items():
        full_path = log_root / ckpt
        if not full_path.exists():
            print(f'  WARNING: {name} checkpoint not found: {full_path}')
            print(f'  Skipping {name}')
            continue
        print(f'  {name}: {full_path} ✓')

    # Per-episode evaluation
    all_episode_metrics = defaultdict(lambda: defaultdict(list))
    summary_metrics = defaultdict(lambda: defaultdict(list))

    for episode in args.episodes:
        print(f'\n{"="*60}')
        print(f'Episode {episode}')
        print(f'{"="*60}')

        episode_dir = os.path.join(args.output_dir, f'episode_{episode:04d}')
        os.makedirs(episode_dir, exist_ok=True)

        model_results = {}

        # Initialize renderers for this episode
        ablation_renderer = None
        baseline_renderer = None

        for model_name, ckpt_path in models.items():
            if 'neural' in model_name.lower() or 'v4b' in model_name.lower():
                try:
                    ablation_renderer = AblationRenderer(
                        renderer_ckpt_path=ckpt_path,
                        cfg=cfg, episode=episode,
                        torch_device=torch_device,
                    )
                except Exception as e:
                    print(f'  [Neural renderer init failed: {e}]')

        # Baseline LBS renderer (independent — uses frozen splats)
        try:
            baseline_renderer = BaselineLBSRenderer(
                cfg=cfg, episode=episode,
                torch_device=torch_device,
            )
        except Exception as e:
            print(f'  [LBS renderer init failed: {e}]')

        renderers = {
            'ablation': ablation_renderer,
            'baseline': baseline_renderer,
        }

        for model_name, ckpt_path in models.items():
            if not (log_root / ckpt_path).exists():
                continue

            print(f'  Rolling out: {model_name}...')
            try:
                pred_positions, gt_positions = rollout_episode(
                    cfg, ckpt_path, episode, torch_device, wp_device
                )
                model_results[model_name] = {
                    'pred_positions': pred_positions,
                    'gt_positions': gt_positions,
                }

                # Compute per-step metrics
                step_metrics = []
                for step in range(len(pred_positions)):
                    pred = pred_positions[step]
                    gt = gt_positions[step]
                    mde = compute_mde(pred, gt)
                    chamfer = compute_chamfer(pred, gt)
                    emd = compute_emd_approx(pred, gt)
                    step_metrics.append({'mde': mde, 'chamfer': chamfer, 'emd': emd})

                all_episode_metrics[model_name][episode] = step_metrics

                # Summary at step 30 (or last available)
                summary_step = min(30, len(step_metrics) - 1)
                m = step_metrics[summary_step]
                summary_metrics[model_name]['mde'].append(m['mde'])
                summary_metrics[model_name]['chamfer'].append(m['chamfer'])
                summary_metrics[model_name]['emd'].append(m['emd'])

                print(f'    Step {summary_step}: MDE={m["mde"]:.6f}, Chamfer={m["chamfer"]:.6f}, EMD={m["emd"]:.6f}')

            except Exception as e:
                print(f'    FAILED: {e}')
                import traceback
                traceback.print_exc()
                continue

        # Precompute baseline LBS renders incrementally through all steps
        baseline_r = renderers.get('baseline')
        if baseline_r is not None and baseline_r.active and hasattr(baseline_r, 'precompute_all_steps'):
            baseline_model_name = [k for k in model_results if 'baseline' in k.lower()]
            if baseline_model_name:
                baseline_preds = model_results[baseline_model_name[0]]['pred_positions']
                baseline_gts = model_results[baseline_model_name[0]]['gt_positions']
                baseline_r.precompute_all_steps(baseline_preds, baseline_gts, args.viz_steps)

        # Save visual comparisons at specified steps
        for step in args.viz_steps:
            if step >= args.max_steps:
                continue
            if not model_results:
                continue

            # Check step is within bounds for all models
            max_available = min(len(d['pred_positions']) for d in model_results.values())
            if step >= max_available:
                print(f'  Skipping viz step {step} (only {max_available} steps available)')
                continue

            panel = make_comparison_panel(
                model_results, step, episode,
                cfg=cfg,
                renderers=renderers,
                K=K, w2c=w2c,
            )
            panel_path = os.path.join(episode_dir, f'comparison_step{step:03d}.jpg')
            cv2.imwrite(panel_path, panel)

        # Save per-episode metric curves
        if model_results:
            ep_metrics = {name: all_episode_metrics[name][episode]
                          for name in model_results if episode in all_episode_metrics[name]}
            if ep_metrics:
                plot_metric_curves(ep_metrics, os.path.join(episode_dir, 'metric_curves.png'))

    # ==========================================================================
    # Summary across all episodes
    # ==========================================================================
    print(f'\n{"="*60}')
    print('SUMMARY (mean across episodes, step 30)')
    print(f'{"="*60}')
    print(f'{"Model":<30} {"MDE":>10} {"Chamfer":>10} {"EMD":>10}')
    print('-' * 62)

    summary_lines = []
    for model_name in models:
        if model_name not in summary_metrics:
            continue
        m = summary_metrics[model_name]
        mde_mean = np.mean(m['mde'])
        chamfer_mean = np.mean(m['chamfer'])
        emd_mean = np.mean(m['emd'])
        line = f'{model_name:<30} {mde_mean:>10.6f} {chamfer_mean:>10.6f} {emd_mean:>10.6f}'
        print(line)
        summary_lines.append(line)

    # Save summary
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write('PGND Ablation Comparison\n')
        f.write(f'Episodes: {args.episodes}\n')
        f.write(f'Max steps: {args.max_steps}\n\n')
        f.write(f'{"Model":<30} {"MDE":>10} {"Chamfer":>10} {"EMD":>10}\n')
        f.write('-' * 62 + '\n')
        for line in summary_lines:
            f.write(line + '\n')

    # Save detailed per-episode per-step CSV
    csv_path = os.path.join(args.output_dir, 'per_episode_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('model,episode,step,mde,chamfer,emd\n')
        for model_name in all_episode_metrics:
            for episode_id in sorted(all_episode_metrics[model_name].keys()):
                step_metrics = all_episode_metrics[model_name][episode_id]
                for step_idx, m in enumerate(step_metrics):
                    f.write(f'{model_name},{episode_id},{step_idx},{m["mde"]:.6f},{m["chamfer"]:.6f},{m["emd"]:.6f}\n')
    print(f'  Saved per-episode metrics: {csv_path}')

    # Plot aggregate metric curves (mean across episodes)
    if all_episode_metrics:
        agg_metrics = {}
        for model_name in all_episode_metrics:
            ep_metrics_list = list(all_episode_metrics[model_name].values())
            if not ep_metrics_list:
                continue
            n_steps = min(len(m) for m in ep_metrics_list)
            agg = []
            for step in range(n_steps):
                mde_vals = [ep[step]['mde'] for ep in ep_metrics_list]
                chamfer_vals = [ep[step]['chamfer'] for ep in ep_metrics_list]
                emd_vals = [ep[step]['emd'] for ep in ep_metrics_list]
                agg.append({
                    'mde': np.mean(mde_vals),
                    'chamfer': np.mean(chamfer_vals),
                    'emd': np.mean(emd_vals),
                })
            agg_metrics[model_name] = agg

        plot_metric_curves(agg_metrics, os.path.join(args.output_dir, 'aggregate_metrics.png'))

    print(f'\nResults saved to: {args.output_dir}')


if __name__ == '__main__':
    main()