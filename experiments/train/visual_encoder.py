"""
visual_encoder.py — DINOv2 Per-Particle Visual Feature Extraction
=================================================================

Extracts per-particle visual features by:
1. Running frozen DINOv2 on camera images → dense feature maps
2. Projecting each particle into camera view(s)
3. Bilinear sampling features at projected 2D locations

Supports multi-camera: features from each visible camera are averaged.

Usage in training loop:
    vis_encoder = VisualEncoder(cfg, camera_ids=[0, 1], device='cuda')
    vis_encoder.setup_episode(episode_name, coord_transform, cam_settings_list)

    # Each step:
    images = [load_image(cam_id, step) for cam_id in camera_ids]
    vis_feat = vis_encoder.extract_per_particle_features(particles, images)
    # vis_feat: (B, N, vis_dim)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """Frozen DINOv2 backbone + per-particle feature sampling."""

    def __init__(
        self,
        model_name: str = 'dinov2_vits14',   # vits14=384d, vitb14=768d
        feature_dim: int = 64,                # output dim after projection
        camera_ids: List[int] = [1],
        image_size: Tuple[int, int] = (480, 848),  # (H, W)
        device: str = 'cuda',
    ):
        super().__init__()
        self.camera_ids = camera_ids
        self.image_h, self.image_w = image_size
        self.device = device

        # Load frozen DINOv2
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino.eval()
        self.dino.requires_grad_(False)
        self.dino.to(device)

        # DINOv2 patch size and raw feature dim
        self.patch_size = 14
        self.dino_dim = self.dino.embed_dim  # 384 for vits14, 768 for vitb14

        # Feature map spatial dimensions (DINOv2 expects multiples of patch_size)
        # We'll resize input to nearest multiple
        self.input_h = (self.image_h // self.patch_size) * self.patch_size  # 476 → 476
        self.input_w = (self.image_w // self.patch_size) * self.patch_size  # 848 → 840
        self.feat_h = self.input_h // self.patch_size  # 34
        self.feat_w = self.input_w // self.patch_size  # 60

        # Learnable projection: dino_dim → feature_dim
        self.proj = nn.Sequential(
            nn.Linear(self.dino_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.proj.to(device)

        self.feature_dim = feature_dim

        # Per-camera calibration (set via setup_episode)
        self._cam_settings: Dict[int, dict] = {}
        self._coord_transform = None

        print(f'[VisualEncoder] DINOv2 {model_name}: '
              f'{self.dino_dim}d → {feature_dim}d, '
              f'feat_map={self.feat_h}x{self.feat_w}, '
              f'cameras={camera_ids}')

    def setup_episode(
        self,
        coord_transform,
        cam_settings_dict: Dict[int, dict],
    ):
        """Set coordinate transform and camera calibration for current episode.

        Args:
            coord_transform: PGNDCoordinateTransform instance
            cam_settings_dict: {camera_id: {'k': intrinsics, 'w2c': 4x4 array, 'w': int, 'h': int}}
        """
        self._coord_transform = coord_transform
        self._cam_settings = cam_settings_dict

    @torch.no_grad()
    def extract_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 on a single image.

        Args:
            image: (3, H, W) float tensor in [0, 1]

        Returns:
            feature_map: (dino_dim, feat_h, feat_w)
        """
        # Resize to DINOv2-compatible size
        img = F.interpolate(
            image.unsqueeze(0),
            size=(self.input_h, self.input_w),
            mode='bilinear', align_corners=False,
        )

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        # DINOv2 forward — get patch tokens
        features = self.dino.forward_features(img)
        patch_tokens = features['x_norm_patchtokens']  # (1, n_patches, dino_dim)

        # Reshape to spatial feature map
        feat_map = patch_tokens[0].reshape(
            self.feat_h, self.feat_w, self.dino_dim
        ).permute(2, 0, 1)  # (dino_dim, feat_h, feat_w)

        return feat_map

    def project_particles_to_camera(
        self,
        particles: torch.Tensor,   # (N, 3) preprocessed space
        camera_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project particles to 2D pixel coordinates in a camera.

        Returns:
            uv: (N, 2) normalized coordinates in [-1, 1] for grid_sample
            visible: (N,) bool mask of particles in front of camera and in frame
        """
        ct = self._coord_transform
        cam = self._cam_settings[camera_id]

        # Preprocessed → world coordinates
        particles_world = ct.inverse_transform(particles)  # (N, 3)

        # World → camera coordinates
        w2c = torch.tensor(cam['w2c'], dtype=torch.float32, device=particles.device)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pts_cam = (R @ particles_world.T + t.unsqueeze(1)).T  # (N, 3)

        # Camera → pixel coordinates
        K = torch.tensor(cam['k'], dtype=torch.float32, device=particles.device)
        pts_2d = (K @ pts_cam.T).T  # (N, 3)
        u = pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)
        v = pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)

        # Normalize to [-1, 1] for grid_sample
        # Feature map coords, not image coords (account for resize)
        u_norm = 2.0 * u / cam['w'] - 1.0
        v_norm = 2.0 * v / cam['h'] - 1.0

        visible = (pts_cam[:, 2] > 0.01) & \
                  (u_norm > -1.0) & (u_norm < 1.0) & \
                  (v_norm > -1.0) & (v_norm < 1.0)

        uv = torch.stack([u_norm, v_norm], dim=-1)  # (N, 2)
        return uv, visible

    def sample_features_at_particles(
        self,
        feature_map: torch.Tensor,  # (dino_dim, feat_h, feat_w)
        uv: torch.Tensor,           # (N, 2) normalized [-1, 1]
        visible: torch.Tensor,      # (N,) bool
    ) -> torch.Tensor:
        """Bilinear sample features at projected particle locations.

        Returns:
            features: (N, dino_dim) — zero for non-visible particles
        """
        N = uv.shape[0]
        dino_dim = feature_map.shape[0]

        # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
        feat = feature_map.unsqueeze(0)  # (1, dino_dim, feat_h, feat_w)
        grid = uv.unsqueeze(0).unsqueeze(1)  # (1, 1, N, 2)

        sampled = F.grid_sample(
            feat, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )  # (1, dino_dim, 1, N)

        sampled = sampled[0, :, 0, :].T  # (N, dino_dim)

        # Zero out features for non-visible particles
        sampled[~visible] = 0.0

        return sampled

    def forward(
        self,
        particles: torch.Tensor,           # (B, N, 3) preprocessed space
        images: Dict[int, torch.Tensor],    # {camera_id: (3, H, W) tensor}
    ) -> torch.Tensor:
        """Extract per-particle visual features from multi-camera images.

        Args:
            particles: (B, N, 3) particle positions in preprocessed space
            images: dict mapping camera_id → (3, H, W) image tensor

        Returns:
            visual_features: (B, N, feature_dim) per-particle features
        """
        B, N, _ = particles.shape
        device = particles.device

        all_features = torch.zeros(B, N, self.dino_dim, device=device)
        visibility_count = torch.zeros(B, N, 1, device=device)

        for cam_id in self.camera_ids:
            if cam_id not in images:
                continue

            # Extract dense feature map
            feat_map = self.extract_feature_map(images[cam_id])  # (dino_dim, fh, fw)

            for b in range(B):
                # Project particles to this camera
                uv, visible = self.project_particles_to_camera(
                    particles[b], cam_id)

                # Sample features
                sampled = self.sample_features_at_particles(
                    feat_map, uv, visible)  # (N, dino_dim)

                all_features[b] += sampled
                visibility_count[b] += visible.float().unsqueeze(-1)

        # Average across cameras (avoid div by zero)
        vis_count_safe = visibility_count.clamp(min=1.0)
        avg_features = all_features / vis_count_safe  # (B, N, dino_dim)

        # Project to output dim
        visual_features = self.proj(avg_features)  # (B, N, feature_dim)

        return visual_features
