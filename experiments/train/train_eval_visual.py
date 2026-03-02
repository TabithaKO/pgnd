"""
train_eval_visual.py — Image-Conditioned PGND Training
=======================================================

Based on train_eval.py (Phase 2 with render loss + curriculum).
Adds DINOv2 per-particle visual conditioning to PGND.

Key differences from train_eval.py:
    - Uses PGNDVisualModel instead of PGNDModel (when use_visual=true)
    - Creates VisualEncoder (frozen DINOv2 + learnable projection)
    - Loads camera images each step, extracts per-particle features
    - Passes vis_feat to model forward()
    - Backward compatible: with use_visual=false behaves identically to train_eval.py

Usage:
    cd ~/pgnd/experiments/train
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python train_eval_visual.py \
        'gpus=[0]' overwrite=True resume=False debug=False \
        train.name=cloth/train_visual_v1 \
        train.source_dataset_name=data/cloth_merged/sub_episodes_v \
        train.dataset_name=cloth/dataset \
        sim.preprocess_scale=0.8 sim.num_grippers=2 \
        train.dataset_load_skip_frame=3 train.dataset_skip_frame=1 \
        train.num_workers=0 train.batch_size=2 \
        train.training_start_episode=162 train.training_end_episode=242 \
        train.eval_start_episode=610 train.eval_end_episode=650 \
        train.num_iterations=100000 \
        train.iteration_eval_interval=5000 train.iteration_log_interval=50 \
        train.iteration_save_interval=5000 train.resume_iteration=0 \
        model.ckpt=cloth/train/ckpt/100000.pt \
        +train.use_visual=true +train.vis_dim=64 \
        +train.vis_model=dinov2_vits14 +train.vis_camera_ids=[1] \
        +train.use_render_loss=true +train.lambda_render=0.1 \
        +train.render_every_n_steps=1 +train.render_camera_id=1 \
        +train.lambda_ssim=0.2 +train.use_mesh_gs=true
"""

from pathlib import Path
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image
import warp as wp
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import kornia
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.material import PGNDModel
from pgnd.data import RealTeleopBatchDataset, RealGripperDataset
from pgnd.utils import Logger, get_root, mkdir

from train.pv_train import do_train_pv
from train.pv_dataset import do_dataset_pv
from train.metric_eval import do_metric

### RENDER LOSS ###
try:
    from train.render_loss import create_render_loss_module
except ImportError:
    create_render_loss_module = None
try:
    from train.render_loss_ablation2 import create_render_loss_module_ablation2
except ImportError:
    create_render_loss_module_ablation2 = None
try:
    from train.render_loss_multicam import create_render_loss_module_ablation2 as create_render_loss_multicam
except ImportError:
    create_render_loss_multicam = None

### VISUAL CONDITIONING ###
try:
    from visual_encoder import VisualEncoder
    from pgnd_visual import PGNDVisualModel
except ImportError:
    try:
        from train.visual_encoder import VisualEncoder
        from train.pgnd_visual import PGNDVisualModel
    except ImportError:
        VisualEncoder = None
        PGNDVisualModel = None
        print('[visual] WARNING: VisualEncoder/PGNDVisualModel not available')

### PHASE 2 VIZ ###
import cv2
import torch.nn.functional as F


def _draw_point_cloud(pts, h, w, color, label, cam_settings=None, coord_transform=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    if cam_settings is not None and coord_transform is not None:
        pts_t = torch.tensor(pts_np, dtype=torch.float32).cuda()
        pts_world = coord_transform.inverse_transform(pts_t).cpu().numpy()
        K = cam_settings['k']; w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]; t_cam = w2c[:3, 3]
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
                cv2.circle(canvas, (u_v[i], v_v[i]), 2, (int(c[0]*255), int(c[1]*255), int(c[2]*255)), -1)
    else:
        mins = pts_np.min(axis=0); maxs = pts_np.max(axis=0)
        span = (maxs - mins).max() + 1e-8; margin = 15
        u = ((pts_np[:, 0] - mins[0]) / span * (w - 2*margin) + margin).astype(int)
        v = ((pts_np[:, 2] - mins[2]) / span * (h - 2*margin - 15) + margin + 15).astype(int)
        for i in range(len(u)):
            if 0 <= u[i] < w and 0 <= v[i] < h:
                cv2.circle(canvas, (u[i], v[i]), 2, color, -1)
    if label:
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return canvas


def _draw_overlay(pred_pts, gt_pts, h, w, cam_settings=None, coord_transform=None):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pred_np = pred_pts.detach().cpu().numpy() if torch.is_tensor(pred_pts) else pred_pts
    gt_np = gt_pts.detach().cpu().numpy() if torch.is_tensor(gt_pts) else gt_pts
    if cam_settings is not None and coord_transform is not None:
        K = cam_settings['k']; w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]; t_cam = w2c[:3, 3]
        def project(p):
            pt = torch.tensor(p, dtype=torch.float32).cuda()
            pw = coord_transform.inverse_transform(pt).cpu().numpy()
            pc = (R_cam @ pw.T + t_cam.reshape(3, 1)).T
            p2 = (K @ pc.T).T
            uu = (p2[:, 0] / (p2[:, 2] + 1e-8) / 848.0 * w).astype(int)
            vv = (p2[:, 1] / (p2[:, 2] + 1e-8) / 480.0 * h).astype(int)
            ok = (pc[:, 2] > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            return uu, vv, ok
        gu, gv, gok = project(gt_np); pu, pv, pok = project(pred_np)
        for i in range(len(gu)):
            if gok[i]: cv2.circle(canvas, (gu[i], gv[i]), 2, (0, 255, 255), -1)
        for i in range(len(pu)):
            if pok[i]: cv2.circle(canvas, (pu[i], pv[i]), 2, (0, 0, 255), -1)
    cv2.putText(canvas, 'GT+Pred Overlay', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, 'GT', (w - 70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(canvas, 'Pred', (w - 35, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return canvas


@torch.no_grad()
def make_training_debug_panel(render_loss_module, pred_particles, gt_particles, step,
                               episode_name, iteration, mde_val=None):
    try:
        from train.render_loss_ablation2 import compute_mesh_from_particles, project_vertex_colors
    except ImportError:
        return None
    rlm = render_loss_module
    if rlm is None or not rlm.active: return None
    cam = rlm.cam_settings; ct = rlm.coord_transform
    if cam is None or ct is None: return None
    col_w, row_h, gap, header_h, row_label_h = 424, 320, 3, 30, 20
    total_w = 3 * col_w + 2 * gap
    total_h = header_h + 2 * (row_label_h + row_h) + gap
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    header = f'iter {iteration} | {episode_name} step {step}'
    if mde_val is not None: header += f' | MDE={mde_val:.4f}'
    cv2.putText(canvas, header, (5, 22), font, 0.6, (255, 255, 255), 1)
    pred = pred_particles[0].detach() if pred_particles.dim() == 3 else pred_particles.detach()
    gt = gt_particles[0].detach() if gt_particles.dim() == 3 else gt_particles.detach()
    def cell_origin(row, col):
        return col * (col_w + gap), header_h + row * (row_label_h + row_h + gap) + row_label_h
    def label_origin(row, col):
        return col * (col_w + gap), header_h + row * (row_label_h + row_h + gap)
    # Row 0: images
    lx, ly = label_origin(0, 0)
    cv2.putText(canvas, 'IMAGES', (lx + 5, ly + 15), font, 0.45, (150, 150, 150), 1)
    cx, cy = cell_origin(0, 0)
    gt_image = rlm.gt_loader.load_frame(step)
    gt_mask = rlm.gt_loader.load_mask(step) if gt_image is not None else None
    gt_rgb = None
    if gt_image is not None:
        gt_np = gt_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        if gt_mask is not None: gt_np = gt_np * gt_mask.detach().cpu().permute(1, 2, 0).numpy()
        gt_rgb = (gt_np * 255).astype(np.uint8)
        canvas[cy:cy+row_h, cx:cx+col_w] = cv2.cvtColor(cv2.resize(gt_rgb, (col_w, row_h)), cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'GT Image', (cx+5, cy+20), font, 0.45, (200, 200, 200), 1)
    try:
        mesh_data = compute_mesh_from_particles(pred.cuda(), method='bpa')
        vertices = pred[:mesh_data.pos.shape[0]].cuda()
        vertex_colors = project_vertex_colors(vertices_preproc=vertices.detach(), image=gt_image, cam_settings=cam, coord_transform=ct)
        rendered_image, _ = rlm.renderer(vertices=vertices, faces=mesh_data.face, vertex_colors=vertex_colors, cam_settings=cam, coord_transform=ct)
        rendered_rgb = (rendered_image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rendered_resized = cv2.resize(rendered_rgb, (col_w, row_h))
        cx1, cy1 = cell_origin(0, 1)
        canvas[cy1:cy1+row_h, cx1:cx1+col_w] = cv2.cvtColor(rendered_resized, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, 'Rendered', (cx1+5, cy1+20), font, 0.45, (200, 200, 200), 1)
        if gt_rgb is not None:
            cx2, cy2 = cell_origin(0, 2)
            diff = np.abs(cv2.cvtColor(rendered_resized, cv2.COLOR_RGB2BGR).astype(float) - cv2.cvtColor(cv2.resize(gt_rgb, (col_w, row_h)), cv2.COLOR_RGB2BGR).astype(float))
            canvas[cy2:cy2+row_h, cx2:cx2+col_w] = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
            cv2.putText(canvas, 'Diff 3x', (cx2+5, cy2+20), font, 0.45, (200, 200, 200), 1)
    except Exception as e:
        cx1, cy1 = cell_origin(0, 1)
        err = np.zeros((row_h, col_w, 3), dtype=np.uint8)
        cv2.putText(err, f'Render err: {str(e)[:50]}', (10, row_h//2), font, 0.4, (0, 0, 255), 1)
        canvas[cy1:cy1+row_h, cx1:cx1+col_w] = err
    # Row 1: point clouds
    lx, ly = label_origin(1, 0)
    cv2.putText(canvas, 'POINT CLOUDS', (lx + 5, ly + 15), font, 0.45, (150, 150, 150), 1)
    cx0, cy0 = cell_origin(1, 0)
    canvas[cy0:cy0+row_h, cx0:cx0+col_w] = _draw_point_cloud(gt, row_h, col_w, (0,255,128), 'GT PC', cam, ct)
    cx1, cy1 = cell_origin(1, 1)
    canvas[cy1:cy1+row_h, cx1:cx1+col_w] = _draw_point_cloud(pred, row_h, col_w, (0,100,255), 'Pred PC', cam, ct)
    cx2, cy2 = cell_origin(1, 2)
    canvas[cy2:cy2+row_h, cx2:cx2+col_w] = _draw_overlay(pred, gt, row_h, col_w, cam, ct)
    if mde_val is not None:
        cv2.putText(canvas, f'MDE={mde_val:.4f}', (cx2+5, cy2+row_h-15), font, 0.5, (255,255,100), 1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class RenderLossCurriculum:
    def __init__(self, schedule=None):
        self.schedule = schedule or [
            (0, 2000, 0.0), (2000, 20000, 0.1), (20000, 50000, 0.3), (50000, 200000, 0.5),
        ]
        self._last_phase = -1
    def get_lambda(self, iteration):
        for i, (start, end, lam) in enumerate(self.schedule):
            if start <= iteration < end:
                if i != self._last_phase:
                    self._last_phase = i
                    print(f'[curriculum] Phase {i}: iter {start}-{end}, lambda_render={lam}')
                return lam
        return self.schedule[-1][2]


root: Path = get_root(__file__)

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        cnt += 1
        for data in dataloader:
            yield data

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


class Trainer:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg, resolve=True))
        wp.init(); wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})
        wp.config.verify_autograd_array_access = True
        gpus = [int(gpu) for gpu in cfg.gpus]
        wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in gpus]
        torch_devices = [torch.device(f'cuda:{gpu}') for gpu in gpus]
        assert len(torch_devices) == 1
        self.wp_device = wp_devices[0]; self.torch_device = torch_devices[0]
        seed = cfg.seed; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.benchmark = True
        log_root: Path = root / 'log'
        exp_root: Path = log_root / cfg.train.name
        mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
        OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)
        ckpt_root: Path = exp_root / 'ckpt'; ckpt_root.mkdir(parents=True, exist_ok=True)
        self.log_root = log_root; self.ckpt_root = ckpt_root
        self.use_pv = cfg.train.use_pv
        self.dataset_non_overwrite = cfg.train.dataset_non_overwrite
        if not self.use_pv: print('not using pv rendering...')
        assert self.cfg.train.source_dataset_name is not None
        self.use_gs = cfg.train.use_gs
        self.verbose = False
        if not cfg.debug:
            if cfg.resume and cfg.train.resume_iteration > 0:
                ckpt_path = ckpt_root / f'{cfg.train.resume_iteration:06d}.pt'
                if ckpt_path.exists():
                    try:
                        ckpt_peek = torch.load(ckpt_path, map_location='cpu')
                        wandb_resume_id = ckpt_peek.get('wandb_run_id', None)
                        del ckpt_peek
                        if wandb_resume_id:
                            print(f'[wandb] Resuming run: {wandb_resume_id}')
                            os.environ['WANDB_RUN_ID'] = wandb_resume_id
                            os.environ['WANDB_RESUME'] = 'must'
                    except Exception as e:
                        print(f'[wandb] Could not load run ID: {e}')
            logger = Logger(cfg, project='pgnd-train')
            self.logger = logger

    def load_train_dataset(self):
        cfg = self.cfg
        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'
        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        assert os.path.exists(source_dataset_root)
        self.dataset = RealTeleopBatchDataset(
            cfg, dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root, device=self.torch_device,
            num_steps=cfg.sim.num_steps_train, train=True,
            dataset_non_overwrite=self.dataset_non_overwrite,
        )
        if cfg.sim.gripper_points:
            self.gripper_dataset = RealGripperDataset(cfg, device=self.torch_device, train=True)

    def init_train(self):
        cfg = self.cfg
        self.dataloader = dataloader_wrapper(
            DataLoader(self.dataset, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True), 'dataset')
        if cfg.sim.gripper_points:
            self.gripper_dataloader = dataloader_wrapper(
                DataLoader(self.gripper_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                           num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True), 'gripper_dataset')

        material_requires_grad = cfg.model.material.requires_grad

        ### VISUAL CONDITIONING ### Model selection
        self.use_visual = getattr(cfg.train, 'use_visual', False)
        self.vis_dim = getattr(cfg.train, 'vis_dim', 64)

        if self.use_visual and PGNDVisualModel is not None:
            print(f'[visual] Using PGNDVisualModel (vis_dim={self.vis_dim})')
            material: nn.Module = PGNDVisualModel(cfg, vis_dim=self.vis_dim)
            material.to(self.torch_device)
            if cfg.model.ckpt:
                baseline_ckpt = torch.load(self.log_root / cfg.model.ckpt, map_location=self.torch_device)
                material.load_baseline_weights(baseline_ckpt['material'])
            elif cfg.resume and cfg.train.resume_iteration > 0:
                ckpt = torch.load(self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt', map_location=self.torch_device)
                material.load_state_dict(ckpt['material'])
        else:
            self.use_visual = False
            material: nn.Module = PGNDModel(cfg)
            material.to(self.torch_device)
            if cfg.resume and cfg.train.resume_iteration > 0:
                assert (self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt').exists()
                ckpt = torch.load(self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt', map_location=self.torch_device)
                material.load_state_dict(ckpt['material'])
            elif cfg.model.ckpt:
                ckpt = torch.load(self.log_root / cfg.model.ckpt, map_location=self.torch_device)
                material.load_state_dict(ckpt['material'])
        ### END VISUAL CONDITIONING ###

        material.requires_grad_(material_requires_grad)
        material.train(True)

        if not (cfg.resume and cfg.train.resume_iteration > 0):
            torch.save({'material': material.state_dict()}, self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt')

        friction: nn.Module = Friction(np.array([cfg.model.friction.value]))
        friction.to(self.torch_device); friction.requires_grad_(False); friction.train(False)

        if material_requires_grad:
            material_optimizer = torch.optim.Adam(material.parameters(), lr=cfg.train.material_lr, weight_decay=cfg.train.material_wd)
            material_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=material_optimizer, T_max=cfg.train.num_iterations)
            if cfg.train.resume_iteration > 0:
                material_lr_scheduler.last_epoch = cfg.train.resume_iteration - 1
                material_lr_scheduler.step()

        criterion = nn.MSELoss(reduction='mean'); criterion.to(self.torch_device)

        total_step_count = cfg.train.resume_iteration * cfg.sim.num_steps_train if (cfg.resume and cfg.train.resume_iteration > 0) else 0

        self.loss_factor_v = cfg.train.loss_factor_v
        self.loss_factor_x = cfg.train.loss_factor_x
        self.material_requires_grad = material_requires_grad
        self.material = material
        self.material_optimizer = material_optimizer
        self.material_lr_scheduler = material_lr_scheduler
        self.criterion = criterion
        self.total_step_count = total_step_count
        self.losses_log = defaultdict(int)
        self.friction = friction

        ### RENDER LOSS ### (same as train_eval.py)
        self.use_render_loss = getattr(cfg.train, 'use_render_loss', False)
        if self.use_render_loss:
            use_mesh_gs = getattr(cfg.train, 'use_mesh_gs', False)
            _camera_ids = getattr(cfg.train, 'render_camera_ids', None)
            if _camera_ids is not None: _camera_ids = list(_camera_ids)

            if use_mesh_gs and _camera_ids is not None and create_render_loss_multicam is not None:
                print(f'[render_loss] MULTICAM (cameras={_camera_ids})')
                self.render_loss_module = create_render_loss_multicam(cfg, self.log_root,
                    lambda_render=getattr(cfg.train, 'lambda_render', 0.1),
                    lambda_ssim=getattr(cfg.train, 'lambda_ssim', 0.2),
                    render_every_n_steps=getattr(cfg.train, 'render_every_n_steps', 2),
                    camera_id=getattr(cfg.train, 'render_camera_id', 1), camera_ids=_camera_ids)
            elif use_mesh_gs and create_render_loss_module_ablation2 is not None:
                print('[render_loss] NEURAL MESH RENDERER')
                self.render_loss_module = create_render_loss_module_ablation2(cfg, self.log_root,
                    lambda_render=getattr(cfg.train, 'lambda_render', 0.1),
                    lambda_ssim=getattr(cfg.train, 'lambda_ssim', 0.2),
                    render_every_n_steps=getattr(cfg.train, 'render_every_n_steps', 2),
                    camera_id=getattr(cfg.train, 'render_camera_id', 1))
            elif create_render_loss_module is not None and not use_mesh_gs:
                self.render_loss_module = create_render_loss_module(cfg, self.log_root,
                    lambda_render=getattr(cfg.train, 'lambda_render', 0.1),
                    lambda_ssim=getattr(cfg.train, 'lambda_ssim', 0.2),
                    lambda_dino=getattr(cfg.train, 'lambda_dino', 0.0),
                    render_every_n_steps=getattr(cfg.train, 'render_every_n_steps', 2),
                    camera_id=getattr(cfg.train, 'render_camera_id', 1))
            else:
                print('[render_loss] WARNING: not available'); self.use_render_loss = False; self.render_loss_module = None

            if self.use_render_loss and self.render_loss_module is not None:
                source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
                all_episodes = sorted(source_dataset_root.glob('episode_*'))
                self._train_episode_names = [ep.name for ep in all_episodes[cfg.train.training_start_episode:cfg.train.training_end_episode]]
                print(f'[render_loss] {len(self._train_episode_names)} episodes')
            else:
                self._train_episode_names = []
        else:
            self.render_loss_module = None; self._train_episode_names = []; print('[render_loss] disabled')

        ### PHASE 2 ###
        self.phase2_mode = getattr(cfg.train, 'phase2_mode', False)
        self.curriculum = None
        if self.phase2_mode and self.render_loss_module is not None:
            pretrained_ckpt = getattr(cfg.train, 'pretrained_renderer_ckpt', None)
            if pretrained_ckpt:
                rp = self.log_root / pretrained_ckpt
                if rp.exists():
                    ckpt = torch.load(str(rp), map_location=self.torch_device)
                    self.render_loss_module.renderer.load_state_dict(ckpt['renderer'])
                    print(f'[phase2] Loaded renderer from {rp}')
            if getattr(cfg.train, 'freeze_renderer', True):
                self.render_loss_module.renderer.requires_grad_(False)
                self.render_loss_module.renderer.eval()
                self.render_loss_module.renderer_optimizer = None
                print('[phase2] Renderer frozen')
            if getattr(cfg.train, 'use_curriculum', True):
                self.curriculum = RenderLossCurriculum(); print('[phase2] Curriculum enabled')

        ### VISUAL CONDITIONING ### Initialize visual encoder
        self.visual_encoder = None
        if self.use_visual and VisualEncoder is not None:
            vis_camera_ids = list(getattr(cfg.train, 'vis_camera_ids', [1]))
            vis_model = getattr(cfg.train, 'vis_model', 'dinov2_vits14')
            self.visual_encoder = VisualEncoder(
                model_name=vis_model, feature_dim=self.vis_dim,
                camera_ids=vis_camera_ids, image_size=(480, 848), device=str(self.torch_device))
            if material_requires_grad:
                self.material_optimizer.add_param_group({
                    'params': self.visual_encoder.proj.parameters(), 'lr': cfg.train.material_lr})
            print(f'[visual] VisualEncoder ready: {vis_model}, cameras={vis_camera_ids}')

    def train(self, start_iteration, end_iteration, save=True):
        cfg = self.cfg
        self.material.train(True)
        for iteration in trange(start_iteration, end_iteration, dynamic_ncols=True):
            if self.material_requires_grad: self.material_optimizer.zero_grad()
            if self.render_loss_module is not None and hasattr(self.render_loss_module, 'renderer_optimizer'):
                if self.render_loss_module.renderer_optimizer is not None: self.render_loss_module.renderer_optimizer.zero_grad()

            losses = defaultdict(int)
            init_state, actions, gt_states = next(self.dataloader)
            x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
            x = x.to(self.torch_device); v = v.to(self.torch_device)
            x_his = x_his.to(self.torch_device); v_his = v_his.to(self.torch_device)
            actions = actions.to(self.torch_device)

            ### RENDER LOSS ### Setup episode
            render_loss_active = False; ep_name = 'unknown'
            if self.render_loss_module is not None:
                try:
                    local_ep_idx = int(episode_vec[0, 0].item()) if episode_vec is not None else 0
                    ep_name = self._train_episode_names[local_ep_idx] if local_ep_idx < len(self._train_episode_names) else f'episode_{local_ep_idx:04d}'
                    render_loss_active = self.render_loss_module.setup_episode(episode_name=ep_name, particles_0=x[0].detach())
                except Exception as e:
                    print(f'[render_loss] setup failed: {e}'); render_loss_active = False

            _current_lambda = getattr(cfg.train, 'lambda_render', 0.1)
            if self.curriculum is not None:
                _current_lambda = self.curriculum.get_lambda(iteration)
                if self.render_loss_module is not None: self.render_loss_module.lambda_render = _current_lambda
                if _current_lambda == 0: render_loss_active = False

            ### VISUAL CONDITIONING ### Setup for this episode
            vis_feat_active = False
            if self.visual_encoder is not None and self.render_loss_module is not None and render_loss_active:
                try:
                    rlm = self.render_loss_module
                    if hasattr(rlm, 'cam_settings') and rlm.cam_settings is not None:
                        cam_dict = getattr(rlm, 'cam_settings_dict', {getattr(cfg.train, 'render_camera_id', 1): rlm.cam_settings})
                        self.visual_encoder.setup_episode(coord_transform=rlm.coord_transform, cam_settings_dict=cam_dict)
                        vis_feat_active = True
                except Exception as e:
                    if iteration < 10: print(f'[visual] Setup failed: {e}')

            if cfg.sim.gripper_points:
                gripper_points, _ = next(self.gripper_dataloader)
                gripper_points = gripper_points.to(self.torch_device)
                gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, actions)

            gt_x, gt_v = gt_states
            gt_x = gt_x.to(self.torch_device); gt_v = gt_v.to(self.torch_device)
            batch_size = gt_x.shape[0]; num_steps_total = gt_x.shape[1]; num_particles = gt_x.shape[2]

            if cfg.sim.gripper_points:
                num_gripper_particles = gripper_x.shape[2]; num_particles_orig = num_particles
                num_particles = num_particles + num_gripper_particles

            sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)
            statics = StaticsBatch(); statics.init(shape=(batch_size, num_particles), device=self.wp_device)
            statics.update_clip_bound(clip_bound); statics.update_enabled(enabled)
            colliders = CollidersBatch()
            num_grippers = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
            colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
            if num_grippers > 0: colliders.initialize_grippers(actions[:, 0])
            enabled = enabled.to(self.torch_device)
            enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

            for step in range(num_steps_total):
                if num_grippers > 0: colliders.update_grippers(actions[:, step])
                x_in = x.clone()
                if step == 0: x_in_gt = x.clone(); v_in_gt = v.clone()
                else: x_in_gt = x_in_gt + v_in_gt * cfg.sim.dt * cfg.sim.interval

                if cfg.sim.gripper_points:
                    x = torch.cat([x, gripper_x[:, step]], dim=1)
                    v = torch.cat([v, gripper_v[:, step]], dim=1)
                    x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                    v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                    if enabled.shape[1] < num_particles:
                        enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                    statics.update_enabled(enabled.cpu())

                ### VISUAL CONDITIONING ### Extract per-particle features
                vis_feat = None
                if vis_feat_active:
                    try:
                        images = {}
                        for cam_id in self.visual_encoder.camera_ids:
                            img = self.render_loss_module.gt_loader.load_frame(step)
                            if img is not None: images[cam_id] = img
                        if images:
                            x_cloth = x[:, :num_particles_orig] if cfg.sim.gripper_points else x
                            vis_feat = self.visual_encoder(x_cloth, images)
                            if cfg.sim.gripper_points:
                                pad = torch.zeros(vis_feat.shape[0], x.shape[1] - num_particles_orig, self.vis_dim, device=vis_feat.device, dtype=vis_feat.dtype)
                                vis_feat = torch.cat([vis_feat, pad], dim=1)
                    except Exception as e:
                        if iteration < 10 and step == 0: print(f'[visual] Feature extraction: {e}')
                        vis_feat = None

                pred = self.material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
                x, v = sim(statics, colliders, step, x, v, self.friction.mu.clone()[None].repeat(batch_size, 1), pred)

                if cfg.sim.gripper_forcing:
                    assert not cfg.sim.gripper_points
                    gripper_xyz = actions[:, step, :, :3]; gripper_v_act = actions[:, step, :, 3:6]
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                    x_gripper_distance = torch.norm(x_from_gripper, dim=-1)
                    x_gripper_distance_mask = (x_gripper_distance < cfg.model.gripper_radius).unsqueeze(-1).repeat(1, 1, 1, 3)
                    gripper_v_expand = gripper_v_act[:, :, None].repeat(1, 1, num_particles, 1)
                    gripper_closed = actions[:, step, :, -1] < 0.5
                    x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))
                    gripper_quat_vel = actions[:, step, :, 10:13]
                    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)
                    grid_from_gripper_axis = x_from_gripper - (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
                    for i in range(gripper_xyz.shape[1]):
                        m = x_gripper_distance_mask[:, i]
                        x[m] = x_in[m] + cfg.sim.dt * gripper_v_expand[:, i][m]
                        v[m] = gripper_v_expand[:, i][m]

                if cfg.sim.n_history > 0:
                    if cfg.sim.gripper_points:
                        x_his = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None].detach()], dim=2).reshape(batch_size, num_particles_orig, -1)
                        v_his = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None].detach()], dim=2).reshape(batch_size, num_particles_orig, -1)
                    else:
                        x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2).reshape(batch_size, num_particles, -1)
                        v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2).reshape(batch_size, num_particles, -1)

                if cfg.sim.gripper_points:
                    x = x[:, :num_particles_orig]; v = v[:, :num_particles_orig]; enabled = enabled[:, :num_particles_orig]

                if self.loss_factor_x > 0:
                    loss_x = self.criterion(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                    losses['loss_x'] += loss_x; self.losses_log['loss_x'] += loss_x.item()
                if self.loss_factor_v > 0:
                    loss_v = self.criterion(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                    losses['loss_v'] += loss_v; self.losses_log['loss_v'] += loss_v.item()

                if render_loss_active:
                    try:
                        result = self.render_loss_module.compute_loss(particles_pred=x, rollout_step=step)
                        render_loss = result[0] if isinstance(result, tuple) else result
                        if render_loss is not None:
                            losses['loss_render'] += render_loss; self.losses_log['loss_render'] += render_loss.item()
                    except Exception as e:
                        print(f'[render_loss] compute failed step {step}: {e}')

                with torch.no_grad():
                    if self.loss_factor_x > 0:
                        loss_x_trivial = self.criterion((x_in_gt + v_in_gt * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                        self.losses_log['loss_x_trivial'] += loss_x_trivial.item()
                    if self.loss_factor_v > 0:
                        loss_v_trivial = self.criterion(v_in_gt[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                        self.losses_log['loss_v_trivial'] += loss_v_trivial.item()
                    loss_x_sanity = self.criterion(x_in[enabled_mask > 0], (x - v * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0]) * self.loss_factor_x
                    self.losses_log['loss_x_sanity'] += loss_x_sanity.item()
                    if step > 0:
                        loss_x_gt_sanity = self.criterion((gt_x[:, step - 1] + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                    else:
                        loss_x_gt_sanity = self.criterion((x_in + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                    self.losses_log['loss_x_gt_sanity'] += loss_x_gt_sanity.item()

                if save and not cfg.debug:
                    self.logger.add_scalar('main/iteration', iteration, step=self.total_step_count)
                    for loss_k, loss_v_val in losses.items():
                        self.logger.add_scalar(f'main/{loss_k}', loss_v_val.item(), step=self.total_step_count)
                self.total_step_count += 1

            loss = sum(losses.values())
            try: loss.backward()
            except Exception as e: print(f'loss.backward() failed: {e}'); continue

            if self.material_requires_grad:
                material_grad_norm = clip_grad_norm_(self.material.parameters(), max_norm=cfg.train.material_grad_max_norm, error_if_nonfinite=True)
                self.material_optimizer.step()

            if render_loss_active and self.render_loss_module is not None:
                if hasattr(self.render_loss_module, 'renderer_optimizer') and self.render_loss_module.renderer_optimizer is not None:
                    self.render_loss_module.renderer_optimizer.step()

            if (iteration + 1) % cfg.train.iteration_log_interval == 0:
                msgs = [cfg.train.name, time.strftime('%H:%M:%S'),
                        'iteration {:{width}d}/{}'.format(iteration + 1, cfg.train.num_iterations, width=len(str(cfg.train.num_iterations))),
                        'pred.norm {:.4f}'.format(pred.norm().item())]
                if self.material_requires_grad:
                    msgs.extend(['e-lr {:.2e}'.format(self.material_optimizer.param_groups[0]['lr']),
                                 'e-|grad| {:.4f}'.format(material_grad_norm)])
                if self.curriculum is not None:
                    msgs.append(f'λ_r {self.curriculum.get_lambda(iteration):.2f}')
                if self.use_visual: msgs.append('vis=ON' if vis_feat_active else 'vis=OFF')
                for loss_k, loss_v_val in self.losses_log.items():
                    msgs.append('{} {:.8f}'.format(loss_k, loss_v_val / cfg.train.iteration_log_interval))
                    if save and not cfg.debug:
                        self.logger.add_scalar(f'stat/mean_{loss_k}', loss_v_val / cfg.train.iteration_log_interval, step=self.total_step_count)
                print('[{}]'.format(','.join(msgs)))
                self.losses_log = defaultdict(int)

            if save and not cfg.debug:
                self.logger.add_scalar('stat/pred_norm', pred.norm().item(), step=self.total_step_count)
            if self.material_requires_grad:
                if save and not cfg.debug:
                    self.logger.add_scalar('stat/material_lr', self.material_optimizer.param_groups[0]['lr'], step=self.total_step_count)
                    self.logger.add_scalar('stat/material_grad_norm', material_grad_norm, step=self.total_step_count)

            if save and (iteration + 1) % cfg.train.iteration_save_interval == 0:
                ckpt_data = {'material': self.material.state_dict()}
                if self.render_loss_module is not None and hasattr(self.render_loss_module, 'renderer'):
                    ckpt_data['renderer'] = self.render_loss_module.renderer.state_dict()
                    if self.render_loss_module.renderer_optimizer is not None:
                        ckpt_data['renderer_optimizer'] = self.render_loss_module.renderer_optimizer.state_dict()
                if self.visual_encoder is not None:
                    ckpt_data['visual_proj'] = self.visual_encoder.proj.state_dict()
                try:
                    import wandb
                    if wandb.run is not None: ckpt_data['wandb_run_id'] = wandb.run.id
                except Exception: pass
                torch.save(ckpt_data, self.ckpt_root / '{:06d}.pt'.format(iteration + 1))

            # Debug panel
            _debug_viz_interval = 500
            if render_loss_active and save and not cfg.debug and (iteration + 1) % _debug_viz_interval == 0 and self.render_loss_module is not None:
                try:
                    import wandb
                    with torch.no_grad():
                        _mde = torch.norm(x[0] - gt_x[:, num_steps_total - 1][0], dim=-1).mean().item()
                    _panel = make_training_debug_panel(self.render_loss_module, x, gt_x[:, num_steps_total - 1],
                        num_steps_total - 1, ep_name, iteration + 1, mde_val=_mde)
                    if _panel is not None:
                        wandb.log({'debug/training_panel': wandb.Image(_panel, caption=f'iter={iteration+1} {ep_name} MDE={_mde:.4f}')}, step=self.total_step_count)
                except Exception as e:
                    if iteration < 2000: print(f'[debug_panel] Error: {e}')

            if self.material_requires_grad: self.material_lr_scheduler.step()

    def eval_episode(self, iteration: int, episode: int, save: bool = True):
        cfg = self.cfg
        log_root: Path = root / 'log'
        eval_name = f'{cfg.train.name}/eval/{cfg.train.dataset_name.split("/")[-1]}/{iteration:06d}'
        exp_root: Path = log_root / eval_name
        if save:
            state_root: Path = exp_root / 'state'
            mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            episode_state_root = state_root / f'episode_{episode:04d}'
            mkdir(episode_state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

        if cfg.train.dataset_name is None: cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'
        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)

        eval_dataset = RealTeleopBatchDataset(cfg, dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root, device=self.torch_device, num_steps=self.cfg.sim.num_steps,
            eval_episode_name=f'episode_{episode:04d}')
        eval_dataloader = dataloader_wrapper(
            DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True), 'dataset')
        if cfg.sim.gripper_points:
            eval_gripper_dataset = RealGripperDataset(cfg, device=self.torch_device)
            eval_gripper_dataloader = dataloader_wrapper(
                DataLoader(eval_gripper_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True), 'gripper_dataset')

        init_state, actions, gt_states, downsample_indices = next(eval_dataloader)
        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
        x = x.to(self.torch_device); v = v.to(self.torch_device)
        x_his = x_his.to(self.torch_device); v_his = v_his.to(self.torch_device)
        actions = actions.to(self.torch_device)

        if cfg.sim.gripper_points:
            gripper_points, _ = next(eval_gripper_dataloader)
            gripper_points = gripper_points.to(self.torch_device)
            gripper_x, gripper_v, gripper_mask = transform_gripper_points(cfg, gripper_points, actions)

        gt_x, gt_v = gt_states
        gt_x = gt_x.to(self.torch_device); gt_v = gt_v.to(self.torch_device)
        batch_size = gt_x.shape[0]; num_steps_total = gt_x.shape[1]; num_particles = gt_x.shape[2]
        assert batch_size == 1

        if cfg.sim.gripper_points:
            num_gripper_particles = gripper_x.shape[2]; num_particles_orig = num_particles
            num_particles = num_particles + num_gripper_particles

        sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)
        statics = StaticsBatch(); statics.init(shape=(batch_size, num_particles), device=self.wp_device)
        statics.update_clip_bound(clip_bound); statics.update_enabled(enabled)
        colliders = CollidersBatch()
        self.material.eval(); self.friction.eval()
        num_grippers = 0 if cfg.sim.gripper_points else cfg.sim.num_grippers
        colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
        if num_grippers > 0: colliders.initialize_grippers(actions[:, 0])
        enabled = enabled.to(self.torch_device)
        enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

        ### VISUAL CONDITIONING ### Setup for eval
        vis_feat_active_eval = False
        _eval_gt_loader = None
        if self.visual_encoder is not None:
            try:
                from render_loss_ablation2 import PGNDCoordinateTransform
                import json
                source_root = self.log_root / str(cfg.train.source_dataset_name)
                ep_path = source_root / f'episode_{episode:04d}'
                meta = np.loadtxt(str(ep_path / 'meta.txt'))
                with open(source_root / 'metadata.json') as f: metadata = json.load(f)
                entry = metadata[episode]; source_dir = Path(entry['path'])
                recording_name = source_dir.parent.name; source_ep_id = int(meta[0])
                n_hist = int(cfg.sim.n_history); lskip = int(cfg.train.dataset_load_skip_frame); dskip = int(cfg.train.dataset_skip_frame)
                frame_start = int(meta[1]) + n_hist * lskip * dskip
                ep_dir = self.log_root / 'data_cloth' / recording_name / f'episode_{source_ep_id:04d}'
                coord_transform = PGNDCoordinateTransform(cfg, ep_path).to_cuda()
                calib_dir = ep_dir / 'calibration'
                intr = np.load(str(calib_dir / 'intrinsics.npy'))
                rvec = np.load(str(calib_dir / 'rvecs.npy')); tvec = np.load(str(calib_dir / 'tvecs.npy'))
                R = cv2.Rodrigues(rvec[1])[0]; t = tvec[1, :, 0]
                c2w = np.eye(4, dtype=np.float64); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
                w2c = np.linalg.inv(c2w).astype(np.float32)
                cam_settings = {'w': 848, 'h': 480, 'k': intr[1], 'w2c': w2c}
                self.visual_encoder.setup_episode(coord_transform=coord_transform, cam_settings_dict={1: cam_settings})
                from train.render_loss import GTImageLoader
                _eval_gt_loader = GTImageLoader(episode_dir=ep_dir, source_frame_start=frame_start,
                    camera_id=1, image_size=(480, 848), skip_frame=lskip * dskip)
                vis_feat_active_eval = True
            except Exception as e:
                print(f'[visual_eval] Setup failed: {e}')

        colliders_save = colliders.export()
        colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
        ckpt = dict(x=x[0], v=v[0], **colliders_save)
        if save: torch.save(ckpt, episode_state_root / f'{0:04d}.pt')

        losses = {}
        with torch.no_grad():
            for step in trange(num_steps_total):
                if num_grippers > 0: colliders.update_grippers(actions[:, step])
                if cfg.sim.gripper_forcing: x_in = x.clone()
                else: x_in = None

                if cfg.sim.gripper_points:
                    x = torch.cat([x, gripper_x[:, step]], dim=1)
                    v = torch.cat([v, gripper_v[:, step]], dim=1)
                    x_his = torch.cat([x_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=x_his.device, dtype=x_his.dtype)], dim=1)
                    v_his = torch.cat([v_his, torch.zeros((gripper_x.shape[0], gripper_x.shape[2], cfg.sim.n_history * 3), device=v_his.device, dtype=v_his.dtype)], dim=1)
                    if enabled.shape[1] < num_particles:
                        enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)
                    statics.update_enabled(enabled.cpu())

                ### VISUAL CONDITIONING ### Per-step eval features
                vis_feat = None
                if vis_feat_active_eval and _eval_gt_loader is not None:
                    try:
                        img = _eval_gt_loader.load_frame(step)
                        if img is not None:
                            x_cloth = x[:, :num_particles_orig] if cfg.sim.gripper_points else x
                            vis_feat = self.visual_encoder(x_cloth, {1: img})
                            if cfg.sim.gripper_points:
                                pad = torch.zeros(vis_feat.shape[0], x.shape[1] - num_particles_orig, self.vis_dim, device=vis_feat.device, dtype=vis_feat.dtype)
                                vis_feat = torch.cat([vis_feat, pad], dim=1)
                    except Exception: vis_feat = None

                pred = self.material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
                if pred.isnan().any(): print('pred isnan'); break
                if pred.isinf().any(): print('pred isinf'); break

                x, v = sim(statics, colliders, step, x, v, self.friction.mu[None].repeat(batch_size, 1), pred)

                if cfg.sim.gripper_forcing:
                    assert not cfg.sim.gripper_points
                    gripper_xyz = actions[:, step, :, :3]; gripper_v_act = actions[:, step, :, 3:6]
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                    x_gripper_distance = torch.norm(x_from_gripper, dim=-1)
                    x_gripper_distance_mask = (x_gripper_distance < cfg.model.gripper_radius).unsqueeze(-1).repeat(1, 1, 1, 3)
                    gripper_v_expand = gripper_v_act[:, :, None].repeat(1, 1, num_particles, 1)
                    gripper_closed = actions[:, step, :, -1] < 0.5
                    x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))
                    gripper_quat_vel = actions[:, step, :, 10:13]
                    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)
                    grid_from_gripper_axis = x_from_gripper - (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
                    for i in range(gripper_xyz.shape[1]):
                        m = x_gripper_distance_mask[:, i]
                        x[m] = x_in[m] + cfg.sim.dt * gripper_v_expand[:, i][m]
                        v[m] = gripper_v_expand[:, i][m]

                if cfg.sim.n_history > 0:
                    if cfg.sim.gripper_points:
                        x_his = torch.cat([x_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], x[:, :num_particles_orig, None].detach()], dim=2).reshape(batch_size, num_particles_orig, -1)
                        v_his = torch.cat([v_his[:, :num_particles_orig].reshape(batch_size, num_particles_orig, -1, 3)[:, :, 1:], v[:, :num_particles_orig, None].detach()], dim=2).reshape(batch_size, num_particles_orig, -1)
                    else:
                        x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2).reshape(batch_size, num_particles, -1)
                        v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2).reshape(batch_size, num_particles, -1)

                if cfg.sim.gripper_points:
                    extra_save = {'gripper_x': gripper_x[0, step], 'gripper_v': gripper_v[0, step], 'gripper_actions': actions[0, step]}
                    x = x[:, :num_particles_orig]; v = v[:, :num_particles_orig]; enabled = enabled[:, :num_particles_orig]
                else:
                    extra_save = {}

                colliders_save = colliders.export()
                colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
                loss_x = nn.functional.mse_loss(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0])
                loss_v = nn.functional.mse_loss(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0])
                losses[step] = dict(loss_x=loss_x.item(), loss_v=loss_v.item())
                ckpt = dict(x=x[0], v=v[0], **colliders_save, **extra_save)
                if save and step % cfg.sim.skip_frame == 0:
                    torch.save(ckpt, episode_state_root / f'{int(step / cfg.sim.skip_frame):04d}.pt')

        metrics = None
        if save:
            for loss_k in losses[0].keys():
                plt.figure(figsize=(10, 5))
                plt.plot([losses[s][loss_k] for s in losses]); plt.title(loss_k); plt.grid()
                plt.savefig(state_root / f'episode_{episode:04d}_{loss_k}.png', dpi=300); plt.close()

            if self.use_pv:
                do_train_pv(cfg, log_root, iteration, [f'episode_{episode:04d}'], eval_dirname='eval',
                    dataset_name=cfg.train.dataset_name.split("/")[-1], eval_postfix='')
            if self.use_gs:
                from .gs import do_gs
                do_gs(cfg, log_root, iteration, [f'episode_{episode:04d}'], eval_dirname='eval',
                    dataset_name=cfg.train.dataset_name.split("/")[-1], eval_postfix='', camera_id=1, with_mask=True, with_bg=True)
            if self.use_pv:
                _ = do_dataset_pv(cfg, log_root / str(cfg.train.dataset_name), [f'episode_{episode:04d}'],
                    save_dir=log_root / f'{cfg.train.name}/eval/{cfg.train.dataset_name.split("/")[-1]}/{iteration:06d}/pv',
                    downsample_indices=downsample_indices)
            metrics = do_metric(cfg, log_root, iteration, [f'episode_{episode:04d}'], downsample_indices,
                eval_dirname='eval', dataset_name=cfg.train.dataset_name.split("/")[-1], eval_postfix='', camera_id=1, use_gs=self.use_gs)
        return metrics

    def eval(self, eval_iteration: int, save: bool = True):
        cfg = self.cfg
        metrics_list = []
        start_episode = cfg.train.eval_start_episode
        end_episode = cfg.train.eval_end_episode if save else cfg.train.eval_start_episode + 2
        for episode in range(start_episode, end_episode):
            metrics = self.eval_episode(eval_iteration, episode, save=save)
            metrics_list.append(metrics)
        if not save: return
        metrics_list = np.array(metrics_list)[:, 0]
        metric_names = ['mde', 'chamfer', 'emd', 'jscore', 'fscore', 'jfscore', 'perception', 'psnr', 'ssim'] if self.use_gs else ['mde', 'chamfer', 'emd']
        median_metric = np.median(metrics_list, axis=0); mean_metric = np.mean(metrics_list, axis=0)
        step_75 = np.percentile(metrics_list, 75, axis=0); step_25 = np.percentile(metrics_list, 25, axis=0)
        std_metric = np.std(metrics_list, axis=0)
        for i, mn in enumerate(metric_names):
            xx = np.arange(1, len(median_metric) + 1)
            plt.figure(figsize=(8, 5)); plt.plot(xx, median_metric[:, i]); plt.xlabel(f"steps, dt={cfg.sim.dt}"); plt.ylabel(mn); plt.grid()
            plt.gca().fill_between(xx, step_25[:, i], step_75[:, i], alpha=0.2)
            save_dir = root / 'log' / cfg.train.name / 'eval' / cfg.train.dataset_name.split("/")[-1] / f'{eval_iteration:06d}' / 'metric'
            plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{i:02d}-{mn}.png'), dpi=300); plt.close()
        if not cfg.debug:
            for i, mn in enumerate(metric_names):
                self.logger.add_scalar(f'metric/{mn}-mean', mean_metric[:, i].mean(), step=self.total_step_count)
                self.logger.add_scalar(f'metric/{mn}-std', std_metric[:, i].mean(), step=self.total_step_count)
                img = np.array(Image.open(os.path.join(save_dir, f'{i:02d}-{mn}.png')).convert('RGB'))
                self.logger.add_image(f'metric_curve/{mn}', img, step=self.total_step_count)

    def test_cuda_mem(self):
        self.init_train()
        self.train(0, 10, save=False)
        self.eval(10, save=False)


@hydra.main(version_base='1.2', config_path=str(root / 'cfg'), config_name='default')
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.load_train_dataset()
    trainer.test_cuda_mem()
    trainer.init_train()
    for iteration in range(cfg.train.resume_iteration, cfg.train.num_iterations, cfg.train.iteration_eval_interval):
        start_iteration = iteration
        end_iteration = min(iteration + cfg.train.iteration_eval_interval, cfg.train.num_iterations)
        trainer.train(start_iteration, end_iteration)
        trainer.eval(end_iteration)


if __name__ == '__main__':
    main()
