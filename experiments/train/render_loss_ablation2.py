"""
render_loss_ablation2.py — Neural Mesh Renderer for Ablation 2 (Corrected)
============================================================================

Uses a LEARNED neural mesh renderer instead of per-face stored GS params.
The MLP predicts Gaussian appearance from local geometric features, making
it topology-agnostic and generalizable across particle configurations.

Key differences from the old version:
    - OLD: MeshGaussianModel with stored per-face params (topology-dependent, blob problem)
    - NEW: NeuralMeshRenderer MLP predicts GS params from geometry + vertex colors
    - Mesh is rebuilt fresh from particles each compute_loss call
    - Vertex colors projected from GT image provide appearance signal
    - MLP weights are the ONLY persistent learned state (shared across all episodes)

Architecture:
    particles (from dynamics)
        ↓
    Build mesh (Delaunay triangulation) — fresh every call
        ↓
    Project vertex colors from GT image
        ↓
    Per-face features: normals, areas, edges, colors
        ↓
    MLP: features → per-Gaussian (color, scale, opacity, bary)
        ↓
    Barycentric interpolation → Gaussian positions
        ↓
    Differentiable rasterization → rendered image
        ↓
    Loss = SSIM + L1 vs GT image → gradients to MLP AND dynamics

At inference:
    Pass any mesh + vertex colors + camera → MLP forward → render RGB
    No per-scene optimization needed.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

# Import the neural mesh renderer
from neural_mesh_renderer import (
    NeuralMeshRenderer,
    MeshFeatureExtractor,
    project_vertex_colors,
    create_neural_mesh_renderer,
)

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

# Import mesh building utility
try:
    from build_cloth_mesh import compute_mesh_from_particles
except ImportError:
    print("[render_loss_ablation2] Warning: Could not import mesh utilities")


# =============================================================================
# Coordinate Transform (unchanged from previous version)
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
        self.cfg = cfg

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
        self.R = self.R.cuda()
        self.translation = self.translation.cuda()
        return self

    def inverse_transform(self, positions_preprocessed: torch.Tensor) -> torch.Tensor:
        """Transform from PGND preprocessed space to original world space.

        Forward: x_preproc = x_world @ R^T * s + t
        Inverse: x_world = ((x_preproc - t) / s) @ R
        """
        positions = (positions_preprocessed - self.translation) / self.scale
        positions = positions @ self.R
        return positions

    def forward_transform(self, positions_world: torch.Tensor) -> torch.Tensor:
        """Transform from world space to PGND preprocessed space."""
        positions = positions_world @ self.R.T
        positions = positions * self.scale
        positions = positions + self.translation
        return positions


# =============================================================================
# Render Loss Module with Neural Mesh Renderer
# =============================================================================

class RenderLossModuleAblation2:
    """Render loss using a neural mesh renderer.

    The neural mesh renderer is an MLP that predicts per-Gaussian appearance
    from local mesh geometry + vertex colors. It generalizes across mesh
    topologies and particle configurations.

    Manages:
        - NeuralMeshRenderer (persistent, learned MLP)
        - Coordinate transforms (per episode)
        - GT image loading (per episode)
        - Mesh building + rendering (per compute_loss call)
        - Loss computation
    """

    def __init__(
        self,
        cfg,
        log_root: Path,
        device: torch.device = torch.device('cuda'),
        lambda_render: float = 0.1,
        lambda_ssim: float = 0.2,
        lambda_dino: float = 0.0,
        lambda_lpips: float = 0.1,
        render_every_n_steps: int = 2,
        camera_id: int = 1,
        image_h: int = 480,
        image_w: int = 848,
        mesh_method: str = 'bpa',
        # Neural renderer params
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        gaussians_per_face: int = 2,
        renderer_lr: float = 1e-4,
    ):
        self.cfg = cfg
        self.log_root = log_root
        self.device = device
        self.lambda_render = lambda_render
        self.lambda_ssim = lambda_ssim
        self.lambda_dino = lambda_dino
        self.lambda_lpips = lambda_lpips
        self.render_every_n_steps = render_every_n_steps
        self.camera_id = camera_id
        self.image_h = image_h
        self.image_w = image_w
        self.mesh_method = mesh_method
        self.renderer_lr = renderer_lr

        self.active = False

        # DINOv2 feature loss (lazy-loaded)
        self.dino_loss = None
        if self.lambda_dino > 0:
            self.dino_loss = DINOv2FeatureLoss.get_instance(device=str(device))

        # Debug image saving
        self._debug_save_counter = 0
        self._debug_save_interval = 100

        # Per-episode state (rebuilt each episode)
        self.coord_transform = None
        self.cam_settings = None
        self.gt_loader = None

        # =====================================================================
        # Create the neural mesh renderer (PERSISTENT across episodes)
        # This is the only learned state — MLP weights
        # =====================================================================
        self.renderer = create_neural_mesh_renderer(
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            gaussians_per_face=gaussians_per_face,
            use_vertex_colors=True,
            device=str(device),
        )

        # Optimizer for the renderer MLP
        self.renderer_optimizer = torch.optim.Adam(
            self.renderer.parameters(),
            lr=renderer_lr,
        )

        # For compatibility with train_eval.py which checks for gs_optimizer
        self.gs_optimizer = None  # Not used — renderer_optimizer is used instead
        self.gs_model = None     # Not used — renderer handles everything

    def setup_episode(
        self,
        episode_name: str,
        particles_0: torch.Tensor,
        force_reload: bool = False,
    ) -> bool:
        """Initialize render loss for an episode.

        Sets up per-episode state: coordinate transforms, camera, GT loader.
        The neural renderer itself persists across episodes.

        Args:
            episode_name: e.g., 'episode_0042'
            particles_0: (N, 3) initial particle positions in PGND preprocessed space
            force_reload: unused (kept for API compatibility)

        Returns:
            True if render loss is active for this episode
        """
        cfg = self.cfg
        self.active = False

        print(f"[RenderLossAblation2] Setting up episode: {episode_name}")

        # =====================================================================
        # Step 1: Resolve paths to episode data
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

        recording_name = source_data_dir.parent.name
        data_cloth_recording = self.log_root / 'data_cloth' / recording_name
        if not data_cloth_recording.exists():
            print(f"  ⚠️  data_cloth recording not found: {data_cloth_recording}")
            return False

        episode_dir = data_cloth_recording / f'episode_{source_episode_id:04d}'

        # =====================================================================
        # Step 2: Setup coordinate transforms (per-episode)
        # =====================================================================

        self.coord_transform = PGNDCoordinateTransform(cfg, episode_data_path).to_cuda()
        print(f"  Coordinate transform initialized")

        # =====================================================================
        # Step 3: Load camera parameters
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
        # Step 4: Setup GT image loader
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
        print(f"  ✅ Render loss active (neural mesh renderer)")
        return True

    def compute_loss(
        self,
        particles_pred: torch.Tensor = None,
        rollout_step: int = 0,
        x_pred: torch.Tensor = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute render loss for a rollout step.

        For each call:
        1. Build mesh fresh from current particles
        2. Project vertex colors from GT image
        3. Run neural renderer MLP to predict Gaussian params
        4. Rasterize to image
        5. Compute loss vs GT

        Gradients flow to:
        - Dynamics model: through particle positions → mesh → Gaussian positions
        - Renderer MLP: through predicted colors/scales/opacities

        Args:
            particles_pred: (bsz, N, 3) predicted particle positions
            rollout_step: current step in rollout
            x_pred: alias for particles_pred

        Returns:
            (loss_render, None): loss_render includes lambda_render scaling
        """
        if particles_pred is None and x_pred is not None:
            particles_pred = x_pred
        if not self.active:
            return None, None

        if rollout_step % self.render_every_n_steps != 0:
            return None, None

        # Load GT image
        gt_image = self.gt_loader.load_frame(rollout_step)
        if gt_image is None:
            return None, None

        # Use first batch element
        particles = particles_pred[0]  # (N, 3) — carries gradient!

        # =====================================================================
        # Step 1: Build mesh from current particles (fresh every call)
        # =====================================================================
        # Detach for mesh topology building (we don't need gradients through
        # Delaunay triangulation), but keep particles with grad for positions
        mesh_data = compute_mesh_from_particles(
            particles.detach(),
            method=self.mesh_method,
        )
        faces = mesh_data.face  # (3, N_faces) — integer indices, no grad needed
        n_verts = mesh_data.pos.shape[0]
        n_faces = faces.shape[1]

        # Use the ORIGINAL particles (with grad) as vertices for position gradients
        # mesh_data.pos may be a subset or reordered — we need to use particles directly
        # since compute_mesh_from_particles should use all particles as vertices
        vertices = particles[:n_verts]  # (N_verts, 3) — WITH gradient

        # =====================================================================
        # Step 2: Project vertex colors from GT image
        # =====================================================================
        vertex_colors = project_vertex_colors(
            vertices_preproc=vertices.detach(),
            image=gt_image,
            cam_settings=self.cam_settings,
            coord_transform=self.coord_transform,
        )  # (N_verts, 3) — no grad (colors are input features, not optimized)

        # =====================================================================
        # Step 3: Neural renderer forward pass
        # =====================================================================
        # MLP predicts per-Gaussian params from face features
        # Positions computed via barycentric interpolation ON vertices (with grad)
        rendered_image, debug_info = self.renderer(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            cam_settings=self.cam_settings,
            coord_transform=self.coord_transform,
        )

        # =====================================================================
        # Step 4: Compute image loss
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

        loss_image = masked_image_loss(
            pred=rendered_image,
            gt=gt_image,
            mask=gt_mask,
            lambda_ssim=self.lambda_ssim,
        )

        # Add LPIPS perceptual loss for high-frequency detail
        if self.lambda_lpips > 0:
            lpips_loss = self.renderer.compute_lpips_loss(
                rendered=rendered_image,
                gt=gt_image,
                mask=gt_mask,
            )
            loss_image = loss_image + self.lambda_lpips * lpips_loss

        # Add DINOv2 loss if enabled
        if self.dino_loss is not None and self.lambda_dino > 0:
            loss_dino = self.dino_loss.compute_loss(
                pred=rendered_image,
                gt=gt_image,
                mask=gt_mask,
            )
            loss_image = (1.0 - self.lambda_dino) * loss_image + self.lambda_dino * loss_dino

        loss_render = loss_image * self.lambda_render

        # =====================================================================
        # Debug visualization
        # =====================================================================
        if self._debug_save_counter % self._debug_save_interval == 0:
            self._save_debug_images(
                rendered_image, gt_image, gt_mask, rollout_step,
                particles=particles, debug_info=debug_info,
            )
        self._debug_save_counter += 1

        return loss_render, None

    def _save_debug_images(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor],
        rollout_step: int,
        particles: torch.Tensor = None,
        debug_info: Dict = None,
    ):
        """Save debug visualization: rendered | GT | diff | projected particles."""
        try:
            with torch.no_grad():
                rendered_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gt_np = gt.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

                if mask is not None:
                    mask_np = mask.detach().cpu().permute(1, 2, 0).numpy()
                    masked_gt_np = gt_np * mask_np
                else:
                    masked_gt_np = gt_np

                diff = np.abs(rendered_np - masked_gt_np)
                diff_amplified = np.clip(diff * 5.0, 0, 1)

                # 4th column: projected particle point cloud
                h, w = rendered_np.shape[:2]
                pc_img = np.zeros((h, w, 3), dtype=np.float32)

                if particles is not None and self.coord_transform is not None:
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

                    depth = pts_cam[:, 2]
                    valid = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

                    if valid.sum() > 0:
                        d_valid = depth[valid]
                        d_min, d_max = d_valid.min(), d_valid.max()
                        d_norm = (d_valid - d_min) / (d_max - d_min + 1e-8)

                        u_v, v_v = u[valid], v[valid]
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

                font = cv2.FONT_HERSHEY_SIMPLEX
                n_g = debug_info.get('n_gaussians', '?') if debug_info else '?'
                n_f = debug_info.get('n_faces', '?') if debug_info else '?'
                cv2.putText(canvas_uint8, f'Rendered ({n_g}G/{n_f}F)', (10, 22), font, 0.5, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'GT (masked)', (w+gap+10, 22), font, 0.5, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'Diff (5x)', (2*w+2*gap+10, 22), font, 0.5, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'Particles', (3*w+3*gap+10, 22), font, 0.5, (255,255,255), 1)

                out_dir = self.log_root / 'render_loss_ablation2_debug'
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / f'debug_{self._debug_save_counter:06d}_step{rollout_step}.jpg'
                cv2.imwrite(str(out_path), cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR))

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'render_debug_ablation2/comparison': wandb.Image(
                                canvas_uint8,
                                caption=f'Neural Mesh | GT | Diff (step {rollout_step})'
                            ),
                        }, commit=False)
                except ImportError:
                    pass
        except Exception as e:
            print(f'[RenderLossAblation2] debug image save failed: {e}')


# =============================================================================
# Factory function
# =============================================================================

def create_render_loss_module_ablation2(cfg, log_root, **kwargs) -> RenderLossModuleAblation2:
    """Factory function to create RenderLossModuleAblation2."""

    defaults = {
        'lambda_render': 0.1,
        'lambda_ssim': 0.2,
        'lambda_dino': 0.0,
        'lambda_lpips': 0.1,
        'render_every_n_steps': 2,
        'camera_id': 1,
        'mesh_method': 'bpa',
        # Neural renderer params
        'hidden_dim': 256,
        'n_hidden_layers': 4,
        'gaussians_per_face': 8,
        'renderer_lr': 1e-4,
    }

    render_cfg = getattr(cfg, 'render_loss_ablation2', None)
    if render_cfg is not None:
        for k in defaults:
            if hasattr(render_cfg, k):
                defaults[k] = getattr(render_cfg, k)

    defaults.update(kwargs)

    return RenderLossModuleAblation2(cfg=cfg, log_root=log_root, **defaults)