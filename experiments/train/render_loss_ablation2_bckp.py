"""
render_loss_ablation2.py — Mesh-Constrained GS Render Loss for Ablation 2
===========================================================================

Differentiable rendering loss for Ablation 2a/2b/2c using mesh-constrained
Gaussian Splatting. Unlike Ablation 1 (which uses LBS with frozen GS params),
this version trains BOTH the dynamics model AND the GS parameters.

Architecture:
    PGND dynamics model predicts particle positions
            ↓
    Particles = mesh vertices (BPA) or control points (Poisson)
            ↓
    Mesh-constrained GS model deforms Gaussians via barycentric interpolation
            ↓
    Differentiable GS rasterization
            ↓
    Loss computation:
        - Dynamics model: 3D geometric loss + image loss (SSIM/DINOv2)
        - GS parameters: image loss only

Key differences from Ablation 1 (render_loss.py):
    - Uses MeshGaussianModel instead of DifferentiableLBS
    - GS parameters are trainable (colors, scales, rotations, opacities, bary_coords)
    - Separate optimizers for dynamics vs GS
    - Support for both BPA (1:1) and Poisson (RBF interpolation) meshes

Usage:
    Place this file in ~/pgnd/experiments/train/render_loss_ablation2.py

    In train_eval.py:
        from render_loss_ablation2 import create_render_loss_module_ablation2

        render_loss = create_render_loss_module_ablation2(cfg, log_root)

        # Per episode setup
        render_loss.setup_episode(episode_name, particles_0)

        # Per rollout step
        loss_render, gs_loss = render_loss.compute_loss(particles_pred, step)

        # Dual backward passes
        loss_dynamics = loss_geo + lambda_render * loss_render
        loss_dynamics.backward()
        dynamics_optimizer.step()

        gs_loss.backward()
        render_loss.gs_optimizer.step()
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

# Import mesh-constrained GS model
from mesh_gaussian_model import MeshGaussianModel

# Import utilities from render_loss.py
try:
    from render_loss import (
        setup_camera_for_render,
        GTImageLoader,
        ssim_loss,
        masked_image_loss,
        DINOv2FeatureLoss,
    )
except ImportError:
    print("[render_loss_ablation2] Warning: Could not import from render_loss.py")
    print("  Some functions will need to be redefined")


# =============================================================================
# Coordinate Transform Utilities
# =============================================================================

class PGNDCoordinateTransform:
    """Handles coordinate transforms between PGND preprocessed space and world space.

    PGND preprocessing applies:
        1. Rotation R (y-up to z-up): x' = R @ x
        2. Scale by preprocess_scale: x'' = s * x'
        3. Translation to center in [0,1]³: x''' = x'' + t

    For rendering, we need to invert this since camera extrinsics are in world space.
    """

    def __init__(self, cfg, episode_data_path: Path):
        """Initialize transforms from episode data.

        Args:
            cfg: PGND config
            episode_data_path: Path to preprocessed episode directory
        """
        self.cfg = cfg

        # Load original trajectory to compute transforms
        traj_path = episode_data_path / 'traj.npz'
        if not traj_path.exists():
            raise FileNotFoundError(f"traj.npz not found: {traj_path}")

        xyz_orig = np.load(str(traj_path))['xyz']  # (T, N, 3)
        xyz = torch.tensor(xyz_orig, dtype=torch.float32)

        # Step 1: Rotation (y-up to z-up)
        self.R = torch.tensor(
            [[1, 0, 0],
             [0, 0, -1],
             [0, 1, 0]],
            dtype=torch.float32
        )
        xyz = torch.einsum('nij,jk->nik', xyz, self.R.T)

        # Step 2: Scale
        self.scale = cfg.sim.preprocess_scale
        xyz = xyz * self.scale

        # Step 3: Translation
        dx = cfg.sim.num_grids[-1]
        if cfg.sim.preprocess_with_table:
            self.translation = torch.tensor([
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ], dtype=xyz.dtype)
        else:
            self.translation = torch.tensor([
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ], dtype=xyz.dtype)

    def to_cuda(self):
        """Move transforms to GPU."""
        self.R = self.R.cuda()
        self.translation = self.translation.cuda()
        return self

    def inverse_transform(self, positions_preprocessed: torch.Tensor) -> torch.Tensor:
        """Transform from PGND preprocessed space to original world space.

        Args:
            positions_preprocessed: (N, 3) positions in [0,1]³ preprocessed space

        Returns:
            (N, 3) positions in original world coordinates
        """
        # Inverse of forward: forward is x_preproc = x_world @ R^T * s + t
        # So inverse is: x_world = ((x_preproc - t) / s) @ R
        positions = (positions_preprocessed - self.translation) / self.scale
        positions = positions @ self.R
        return positions

    def forward_transform(self, positions_world: torch.Tensor) -> torch.Tensor:
        """Transform from world space to PGND preprocessed space.

        Args:
            positions_world: (N, 3) positions in original world coordinates

        Returns:
            (N, 3) positions in [0,1]³ preprocessed space
        """
        # Forward: x_preproc = R @ x_world * s + t
        positions = positions_world @ self.R
        positions = positions * self.scale
        positions = positions + self.translation
        return positions


# =============================================================================
# Mesh-Based Render Loss Module
# =============================================================================

class RenderLossModuleAblation2:
    """Render loss module for Ablation 2 with mesh-constrained Gaussian Splatting.

    Manages:
        - MeshGaussianModel (per episode)
        - Coordinate transforms (preprocessed ↔ world)
        - GT image loading
        - Differentiable rendering
        - Loss computation (SSIM, DINOv2)
        - Separate optimizers for dynamics vs GS
    """

    def __init__(
        self,
        cfg,
        log_root: Path,
        device: torch.device = torch.device('cuda'),
        lambda_render: float = 0.1,
        lambda_ssim: float = 0.2,
        lambda_dino: float = 0.0,
        render_every_n_steps: int = 2,
        camera_id: int = 1,
        image_h: int = 480,
        image_w: int = 848,
        mesh_method: str = 'bpa',
        opacity_threshold: float = 0.1,
        # GS optimizer params
        lr_position: float = 1e-4,
        lr_color: float = 2.5e-3,
        lr_scale: float = 5e-3,
        lr_rotation: float = 1e-3,
        lr_opacity: float = 5e-2,
    ):
        self.cfg = cfg
        self.log_root = log_root
        self.device = device
        self.lambda_render = lambda_render
        self.lambda_ssim = lambda_ssim
        self.lambda_dino = lambda_dino
        self.render_every_n_steps = render_every_n_steps
        self.camera_id = camera_id
        self.image_h = image_h
        self.image_w = image_w
        self.mesh_method = mesh_method
        self.opacity_threshold = opacity_threshold

        # GS optimizer learning rates
        self.lr_position = lr_position
        self.lr_color = lr_color
        self.lr_scale = lr_scale
        self.lr_rotation = lr_rotation
        self.lr_opacity = lr_opacity

        self.active = False

        # DINOv2 feature loss (lazy-loaded)
        self.dino_loss = None
        if self.lambda_dino > 0:
            self.dino_loss = DINOv2FeatureLoss.get_instance(device=str(device))

        # Debug image saving
        self._debug_save_counter = 0
        self._debug_save_interval = 100

        # Per-episode state
        self.gs_model = None
        self.gs_optimizer = None
        self.coord_transform = None
        self.cam_settings = None
        self.gt_loader = None

    def setup_episode(
        self,
        episode_name: str,
        particles_0: torch.Tensor,
        force_reload: bool = False,
    ) -> bool:
        """Initialize render loss for an episode.

        Args:
            episode_name: e.g., 'episode_0042'
            particles_0: (N, 3) initial particle positions in PGND preprocessed space
            force_reload: If True, rebuild GS model even if checkpoint exists

        Returns:
            True if GS data is available and render loss is active
        """
        cfg = self.cfg
        self.active = False

        print(f"[RenderLossAblation2] Setting up episode: {episode_name}")

        # =====================================================================
        # Step 1: Resolve paths to episode data and .splat file
        # =====================================================================

        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        episode_data_path = source_dataset_root / episode_name
        meta_path = episode_data_path / 'meta.txt'

        if not meta_path.exists():
            print(f"  ⚠️  meta.txt not found: {meta_path}")
            return False

        meta = np.loadtxt(str(meta_path))
        source_episode_id = int(meta[0])
        n_history = int(cfg.sim.n_history)
        load_skip = int(cfg.train.dataset_load_skip_frame)
        ds_skip = int(cfg.train.dataset_skip_frame)
        source_frame_start = int(meta[1]) + n_history * load_skip * ds_skip

        # Load metadata.json to find source recording
        metadata_path = source_dataset_root / 'metadata.json'
        if not metadata_path.exists():
            print(f"  ⚠️  metadata.json not found")
            return False

        import json
        with open(metadata_path) as f:
            metadata = json.load(f)

        episode_idx = int(episode_name.split('_')[1])
        if isinstance(metadata, list):
            if episode_idx >= len(metadata):
                return False
            entry = metadata[episode_idx]
        else:
            entry = metadata.get(str(episode_idx), metadata.get(episode_name))

        if entry is None:
            return False

        if isinstance(entry, dict):
            source_data_dir = Path(entry['path'])
        else:
            source_data_dir = Path(str(entry))

        # Extract recording name
        recording_name = source_data_dir.parent.name

        # GS data path
        data_cloth_recording = self.log_root / 'data_cloth' / recording_name
        if not data_cloth_recording.exists():
            print(f"  ⚠️  data_cloth recording not found: {data_cloth_recording}")
            return False

        episode_dir = data_cloth_recording / f'episode_{source_episode_id:04d}'
        gs_dir = episode_dir / 'gs'

        if not gs_dir.exists():
            print(f"  ⚠️  .splat directory not found: {gs_dir}")
            return False

        # Find closest splat to source_frame_start
        splat_files = sorted(gs_dir.glob('*.splat'))
        if not splat_files:
            print(f"  ⚠️  No .splat files found")
            return False

        frame_nums = [int(f.stem) for f in splat_files]
        closest_frame = min(frame_nums, key=lambda x: abs(x - source_frame_start))
        gs_path = gs_dir / f'{closest_frame:06d}.splat'

        print(f"  Source: {recording_name}/episode_{source_episode_id:04d}")
        print(f"  Splat: {gs_path.name} (frame {closest_frame}, target {source_frame_start})")

        # =====================================================================
        # Step 2: Initialize or reuse MeshGaussianModel
        # =====================================================================
        # Same cloth across all episodes → share one GS model.
        # First call: build mesh from particles, init Gaussians randomly.
        # Subsequent calls: reuse existing GS model (keeps learned colors/scales/etc).

        if self.gs_model is None:
            print(f"  Initializing GS model from mesh (cloth-splatting style)")
            self.gs_model = MeshGaussianModel(sh_degree=0)
            self.gs_model.initialize_from_mesh(
                particles=particles_0,
                method=self.mesh_method,
                gaussian_init_factor=2,
                spatial_lr_scale=1.0,
                device=str(self.device),
            )
            print(f"  Created shared GS model (will persist across episodes)")

            # Initialize colors from GT image projection
            try:
                self._init_colors_from_projection(
                    particles_0, episode_data_path, episode_dir,
                    source_frame_start, load_skip * ds_skip
                )
            except Exception as e:
                print(f"  ⚠️ Color init from projection failed: {e}")
                print(f"  Falling back to mid-gray init")
                self.gs_model._colors.data[:, 0, :] = 0.5
        else:
            print(f"  Reusing shared GS model ({self.gs_model.num_gaussians} Gaussians)")

        # =====================================================================
        # Step 3: Setup GS optimizer (only on first call)
        # =====================================================================

        if self.gs_optimizer is None:
            param_groups = self.gs_model.get_optimizer_param_groups(
                lr_position=self.lr_position,
                lr_color=self.lr_color,
                lr_scale=self.lr_scale,
                lr_rotation=self.lr_rotation,
                lr_opacity=self.lr_opacity,
            )
            self.gs_optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
            print(f"  GS optimizer: {len(param_groups)} parameter groups")
        else:
            print(f"  Reusing GS optimizer")

        # =====================================================================
        # Step 4: Setup coordinate transforms
        # =====================================================================

        self.coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()
        print(f"  Coordinate transform initialized")

        # =====================================================================
        # Step 5: Load camera parameters
        # =====================================================================

        calib_dir = episode_dir / 'calibration'
        if not calib_dir.exists():
            print(f"  ⚠️  Calibration not found: {calib_dir}")
            return False

        intr = np.load(str(calib_dir / 'intrinsics.npy'))
        rvec = np.load(str(calib_dir / 'rvecs.npy'))
        tvec = np.load(str(calib_dir / 'tvecs.npy'))

        R = cv2.Rodrigues(rvec[self.camera_id])[0]
        t = tvec[self.camera_id, :, 0]

        # w2c matrix
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        w2c = np.linalg.inv(c2w).astype(np.float32)

        self.cam_settings = {
            'w': self.image_w,
            'h': self.image_h,
            'k': intr[self.camera_id],
            'w2c': w2c,
        }

        print(f"  Camera {self.camera_id} loaded")

        # =====================================================================
        # Step 6: Setup GT image loader
        # =====================================================================

        try:
            self.gt_loader = GTImageLoader(
                episode_dir=episode_dir,
                source_frame_start=source_frame_start,
                camera_id=self.camera_id,
                image_size=(self.image_h, self.image_w),
                skip_frame=load_skip * ds_skip,
            )
            print(f"  GT image loader initialized")
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            return False

        self.active = True
        print(f"  ✅ Render loss active!")
        return True

    def _init_colors_from_projection(
        self, particles_0, episode_data_path, episode_dir,
        source_frame_start, skip_frame
    ):
        """Project particles onto GT image to get initial colors."""
        import cv2 as cv

        # Get vertices in world space
        coord_transform = PGNDCoordinateTransform(self.cfg, episode_data_path).to_cuda()
        vertices_world = coord_transform.inverse_transform(particles_0)

        # Get Gaussian world positions via mesh
        gs_world = coord_transform.inverse_transform(
            self.gs_model.get_xyz(deformed_vertices=particles_0)
        )

        # Load camera
        calib_dir = episode_dir / 'calibration'
        intr = np.load(str(calib_dir / 'intrinsics.npy'))
        rvec = np.load(str(calib_dir / 'rvecs.npy'))
        tvec = np.load(str(calib_dir / 'tvecs.npy'))

        R = cv.Rodrigues(rvec[self.camera_id])[0]
        t = tvec[self.camera_id, :, 0]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        w2c = np.linalg.inv(c2w).astype(np.float32)

        # Load GT image at frame 0
        gt_loader = GTImageLoader(
            episode_dir=episode_dir,
            source_frame_start=source_frame_start,
            camera_id=self.camera_id,
            image_size=(self.image_h, self.image_w),
            skip_frame=skip_frame,
        )
        gt_image = gt_loader.load_frame(0)
        if gt_image is None:
            raise ValueError("Could not load GT image for color init")

        self.gs_model.initialize_colors_from_image(
            vertices_world=particles_0,
            image=gt_image,
            intrinsics=intr[self.camera_id],
            w2c=w2c,
        )
        print(f"  ✅ Colors initialized from GT image projection")

    def compute_loss(
        self,
        particles_pred: torch.Tensor = None,
        rollout_step: int = 0,
        x_pred: torch.Tensor = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute render loss for a rollout step.

        Args:
            particles_pred: (bsz, N, 3) predicted particle positions in PGND preprocessed space
            rollout_step: current step in rollout

        Returns:
            (loss_render, loss_gs):
                - loss_render: Scalar loss for dynamics model (scaled by lambda_render)
                - loss_gs: Scalar loss for GS parameters (unscaled)
            Both are None if skipping this step
        """
        # Accept x_pred as alias for particles_pred (compatibility with ablation 1 call site)
        if particles_pred is None and x_pred is not None:
            particles_pred = x_pred
        if not self.active:
            return None, None

        # Only render every N steps
        if rollout_step % self.render_every_n_steps != 0:
            return None, None

        # Load GT image
        gt_image = self.gt_loader.load_frame(rollout_step)
        if gt_image is None:
            return None, None

        # Use first batch element
        particles = particles_pred[0]  # (N, 3) — carries gradient!

        # =====================================================================
        # Forward pass: mesh deformation → Gaussian positions
        # =====================================================================

        # Gaussians are deformed via barycentric interpolation
        # particles are mesh vertices (BPA) or control points (Poisson)
        gaussian_positions_preproc = self.gs_model.get_xyz(deformed_vertices=particles)
        gaussian_rotations = self.gs_model.get_rotation(deformed_vertices=particles)

        # Transform to world space for rendering (camera is in world space)
        gaussian_positions_world = self.coord_transform.inverse_transform(gaussian_positions_preproc)

        # Get other GS parameters
        gaussian_colors = self.gs_model.get_colors()
        gaussian_scales = self.gs_model.get_scales()
        gaussian_opacities = self.gs_model.get_opacities()

        # =====================================================================
        # Render with differentiable GS rasterizer
        # =====================================================================

        render_data = {
            'means3D': gaussian_positions_world,
            'colors_precomp': gaussian_colors,
            'rotations': F.normalize(gaussian_rotations, dim=-1),
            'opacities': gaussian_opacities,
            'scales': gaussian_scales,
            'means2D': torch.zeros_like(gaussian_positions_world, requires_grad=True, device="cuda") + 0,
        }

        cam = setup_camera_for_render(
            w=self.cam_settings['w'],
            h=self.cam_settings['h'],
            k=self.cam_settings['k'],
            w2c=self.cam_settings['w2c'],
        )

        rendered_image, _, _ = GaussianRasterizer(raster_settings=cam)(**render_data)
        # rendered_image: (3, H, W) with gradients to:
        #   - gaussian_positions_world → gaussian_positions_preproc → particles (dynamics)
        #   - gaussian_colors, gaussian_scales, gaussian_rotations, gaussian_opacities (GS params)

        # =====================================================================
        # Load mask and compute image loss
        # =====================================================================

        gt_mask = self.gt_loader.load_mask(rollout_step)

        # Resize if needed
        if gt_image.shape[-2:] != rendered_image.shape[-2:]:
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=rendered_image.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(0)

            if gt_mask is not None:
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0),
                    size=rendered_image.shape[-2:],
                    mode='nearest',
                ).squeeze(0)

        # Compute image loss (SSIM + L1)
        loss_image = masked_image_loss(
            pred=rendered_image,
            gt=gt_image,
            mask=gt_mask,
            lambda_ssim=self.lambda_ssim,
        )

        # Add DINOv2 loss if enabled
        if self.dino_loss is not None and self.lambda_dino > 0:
            loss_dino = self.dino_loss.compute_loss(
                pred=rendered_image,
                gt=gt_image,
                mask=gt_mask,
            )
            loss_image = (1.0 - self.lambda_dino) * loss_image + self.lambda_dino * loss_dino

        # =====================================================================
        # Split loss for dual backward passes
        # =====================================================================

        # loss_render: scaled loss for dynamics model (backprop through positions)
        # loss_gs: unscaled loss for GS parameters (backprop through colors/scales/etc)
        # Both come from the same loss_image but will be backward()'d separately

        # Return loss for joint backward:
        # loss.backward() in train_eval.py will send gradients to BOTH
        # dynamics model (through particles) AND GS model (through colors/scales/etc)
        # Then both optimizers step after backward.
        loss_render = loss_image * self.lambda_render
        loss_gs = None  # GS optimizer steps after loss.backward() in train_eval.py

        # =====================================================================
        # Debug visualization
        # =====================================================================

        if self._debug_save_counter % self._debug_save_interval == 0:
            self._save_debug_images(rendered_image, gt_image, gt_mask, rollout_step, particles=particles)
        self._debug_save_counter += 1

        return loss_render, loss_gs

    def _save_debug_images(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor],
        rollout_step: int,
        particles: torch.Tensor = None,
    ):
        """Save debug visualization with 4 columns: rendered | GT | diff | projected particles."""
        try:
            with torch.no_grad():
                rendered_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gt_np = gt.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

                if mask is not None:
                    mask_np = mask.detach().cpu().permute(1, 2, 0).numpy()
                    masked_gt_np = gt_np * mask_np
                else:
                    masked_gt_np = gt_np

                # Difference
                diff = np.abs(rendered_np - masked_gt_np)
                diff_amplified = np.clip(diff * 5.0, 0, 1)

                # 4th column: projected particle point cloud
                h, w = rendered_np.shape[:2]
                pc_img = np.zeros((h, w, 3), dtype=np.float32)

                if particles is not None and self.coord_transform is not None and self.cam_settings is not None:
                    pts_world = self.coord_transform.inverse_transform(particles.detach())
                    pts_np = pts_world.cpu().numpy()

                    K = self.cam_settings['k']
                    w2c = self.cam_settings['w2c']
                    R_cam = w2c[:3, :3]
                    t_cam = w2c[:3, 3]

                    pts_cam = (R_cam @ pts_np.T + t_cam.reshape(3, 1)).T
                    pts_2d = (K @ pts_cam.T).T
                    u = (pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)).astype(int)
                    v = (pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)).astype(int)

                    # Depth for coloring (normalize to [0,1])
                    depth = pts_cam[:, 2]
                    valid = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

                    if valid.sum() > 0:
                        d_valid = depth[valid]
                        d_min, d_max = d_valid.min(), d_valid.max()
                        d_norm = (d_valid - d_min) / (d_max - d_min + 1e-8)

                        u_v, v_v = u[valid], v[valid]
                        # Draw particles as small circles, colored by depth
                        for i in range(len(u_v)):
                            color = (1.0 - d_norm[i]) * np.array([0.2, 0.8, 1.0]) + d_norm[i] * np.array([1.0, 0.3, 0.1])
                            cv2.circle(pc_img, (u_v[i], v_v[i]), 2, color.tolist(), -1)

                # Side-by-side (4 columns)
                gap = 2
                canvas = np.zeros((h + 30, w * 4 + gap * 3, 3), dtype=np.float32)

                y0 = 30
                canvas[y0:y0+h, 0:w] = rendered_np
                canvas[y0:y0+h, w+gap:2*w+gap] = masked_gt_np
                canvas[y0:y0+h, 2*w+2*gap:3*w+2*gap] = diff_amplified
                canvas[y0:y0+h, 3*w+3*gap:4*w+3*gap] = pc_img

                canvas_uint8 = (canvas * 255).astype(np.uint8)

                # Labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas_uint8, 'Rendered (mesh)', (10, 22), font, 0.6, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'GT (masked)', (w+gap+10, 22), font, 0.6, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'Diff (5x)', (2*w+2*gap+10, 22), font, 0.6, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'Particles', (3*w+3*gap+10, 22), font, 0.6, (255,255,255), 1)

                out_dir = self.log_root / 'render_loss_ablation2_debug'
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / f'debug_{self._debug_save_counter:06d}_step{rollout_step}.jpg'
                cv2.imwrite(str(out_path), cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR))

                # Try W&B logging
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'render_debug_ablation2/comparison': wandb.Image(
                                canvas_uint8,
                                caption=f'Mesh-GS | GT | Diff (step {rollout_step})'
                            ),
                        }, commit=False)
                except ImportError:
                    pass
        except Exception as e:
            print(f'[RenderLossAblation2] debug image save failed: {e}')

    def save_gs_checkpoint(self, episode_name: str):
        """Save current GS model checkpoint."""
        if self.gs_model is None:
            return

        checkpoint_dir = self.log_root / 'gs_checkpoints' / episode_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gs_model.save(checkpoint_dir)
        print(f"[RenderLossAblation2] Saved GS checkpoint: {checkpoint_dir}")


# =============================================================================
# Factory function
# =============================================================================

def create_render_loss_module_ablation2(cfg, log_root, **kwargs) -> RenderLossModuleAblation2:
    """Factory function to create RenderLossModuleAblation2 with config defaults.

    Override defaults via kwargs or by adding to PGND config:
        render_loss_ablation2:
            lambda_render: 0.1
            lambda_ssim: 0.2
            lambda_dino: 0.0
            render_every_n_steps: 2
            mesh_method: 'bpa'  # or 'poisson'
    """
    # Pull from config if available
    render_cfg = getattr(cfg, 'render_loss_ablation2', None)

    defaults = {
        'lambda_render': 0.1,
        'lambda_ssim': 0.2,
        'lambda_dino': 0.0,
        'render_every_n_steps': 2,
        'camera_id': 1,
        'mesh_method': 'bpa',
        'opacity_threshold': 0.1,
        # GS optimizer params
        'lr_position': 1e-4,
        'lr_color': 2.5e-3,
        'lr_scale': 5e-3,
        'lr_rotation': 1e-3,
        'lr_opacity': 5e-2,
    }

    if render_cfg is not None:
        for k in defaults:
            if hasattr(render_cfg, k):
                defaults[k] = getattr(render_cfg, k)

    defaults.update(kwargs)

    return RenderLossModuleAblation2(cfg=cfg, log_root=log_root, **defaults)
