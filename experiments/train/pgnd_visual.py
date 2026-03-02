"""
pgnd_visual.py — Image-Conditioned PGND Dynamics Model
=======================================================

Drop-in replacement for PGNDModel that accepts optional per-particle
visual features from DINOv2. When visual features are provided, they
are concatenated with the geometric features before the PointNet encoder.

When no visual features are provided (vis_feat=None), the model behaves
identically to the original PGNDModel — so it's backward compatible.

The only change to the PointNet input channel dim:
    Original:  channel = 6 * (1 + n_history)
    Visual:    channel = 6 * (1 + n_history) + vis_dim

Usage:
    model = PGNDVisualModel(cfg, vis_dim=64)
    # With visual features:
    pred = model(x, v, x_his, v_his, enabled, vis_feat=visual_features)
    # Without (backward compatible):
    pred = model(x, v, x_his, v_his, enabled)
"""

from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Import original network components
import sys
from pathlib import Path
_pgnd_root = Path(__file__).resolve().parent.parent.parent / 'pgnd'
if str(_pgnd_root.parent) not in sys.path:
    sys.path.insert(0, str(_pgnd_root.parent))

from pgnd.material.network.pointnet import PointNetEncoder
from pgnd.material.network.nerf import CondNeRFModel
from pgnd.material.network.utils import get_grid_locations, fill_grid_locations


class PGNDVisualModel(nn.Module):
    """PGND dynamics model with optional per-particle visual conditioning.

    Architecture identical to PGNDModel except:
    - PointNet input channel is extended by vis_dim when visual features present
    - A separate PointNet encoder handles the extended input
    - Everything else (grid aggregation, CondNeRF decoder, etc.) is unchanged
    """

    def __init__(self, cfg: DictConfig, vis_dim: int = 0) -> None:
        super().__init__()

        self.feature_dim = 64
        self.radius = cfg.model.material.radius
        self.n_history = cfg.sim.n_history
        self.num_grids_list = cfg.sim.num_grids[:3]
        self.dx = cfg.sim.num_grids[3]
        self.inv_dx = 1 / self.dx
        self.requires_grad = cfg.model.material.requires_grad
        self.pe_num_func = int(np.log2(self.inv_dx)) + cfg.model.material.pe_num_func_res
        self.pe_dim = 3 + self.pe_num_func * 6
        self.output_scale = cfg.model.material.output_scale
        self.input_scale = cfg.model.material.input_scale
        self.absolute_y = cfg.model.material.absolute_y

        self.vis_dim = vis_dim
        self.geom_channel = 6 * (1 + self.n_history)

        # Original geometry-only encoder (always present for backward compat)
        self.encoder = PointNetEncoder(
            global_feat=(cfg.model.material.radius <= 0),
            feature_transform=False,
            feature_dim=self.feature_dim,
            channel=self.geom_channel,
        )

        # Visual-conditioned encoder (separate weights, used when vis_feat given)
        if vis_dim > 0:
            self.encoder_visual = PointNetEncoder(
                global_feat=(cfg.model.material.radius <= 0),
                feature_transform=False,
                feature_dim=self.feature_dim,
                channel=self.geom_channel + vis_dim,
            )
        else:
            self.encoder_visual = None

        # Decoder is shared — it always receives feature_dim conditioning
        self.decoder = CondNeRFModel(
            xyz_dim=self.pe_dim,
            condition_dim=self.feature_dim,
            out_channel=3,
            num_layers=2,
            hidden_size=64,
            skip_connect_every=4,
        )

    def positional_encoding(self, tensor):
        num_encoding_functions = self.pe_num_func
        if num_encoding_functions == 0:
            return tensor

        encoding = [tensor]
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    def load_baseline_weights(self, state_dict):
        """Load weights from a standard PGNDModel checkpoint.

        Maps encoder → encoder (geometry-only path stays intact).
        Decoder weights load directly.
        Visual encoder starts from scratch (or can be initialized from encoder).
        """
        # Load geometry encoder + decoder
        own_state = self.state_dict()
        loaded = 0
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1

        # Optionally initialize visual encoder from geometry encoder
        if self.encoder_visual is not None:
            geom_state = {k: v for k, v in state_dict.items() if k.startswith('encoder.')}
            vis_state = self.encoder_visual.state_dict()
            initialized = 0
            for name, param in geom_state.items():
                vis_name = name.replace('encoder.', '')
                if vis_name in vis_state:
                    # Only copy if shapes match (they won't for conv1/stn first layer)
                    if vis_state[vis_name].shape == param.shape:
                        vis_state[vis_name].copy_(param)
                        initialized += 1
            self.encoder_visual.load_state_dict(vis_state)
            print(f'[PGNDVisualModel] Loaded {loaded} params from baseline, '
                  f'initialized {initialized} visual encoder params from geometry encoder')
        else:
            print(f'[PGNDVisualModel] Loaded {loaded} params from baseline (no visual encoder)')

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        x_his: Tensor,
        v_his: Tensor,
        enabled: Tensor,
        vis_feat: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: (B, N, 3) particle positions
            v: (B, N, 3) particle velocities
            x_his: (B, N, n_history*3) position history
            v_his: (B, N, n_history*3) velocity history
            enabled: (B, N) particle mask
            vis_feat: (B, N, vis_dim) optional per-particle visual features

        Returns:
            grid_pred: full grid prediction tensor (same as PGNDModel)
        """
        bsz = x.shape[0]
        num_particles = x.shape[1]
        v = v * self.input_scale
        v_his = v_his * self.input_scale

        x_his = x_his.reshape(bsz, num_particles, self.n_history, 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history, 3)
        x_his = x_his.detach()
        v_his = v_his.detach()

        x_grid, grid_idxs = get_grid_locations(x, self.num_grids_list, self.dx)
        x_grid = x_grid.detach()
        grid_idxs = grid_idxs.detach()

        # Centering
        x_center = x.mean(1, keepdim=True)
        if self.absolute_y:
            x_center[:, :, 1] = 0
        x = x - x_center
        x_his = x_his - x_center[:, :, None]

        # Random azimuth rotation (training only)
        if self.training:
            theta = torch.rand(bsz, 1, device=x.device) * 2 * np.pi
            rot = torch.stack([
                torch.cos(theta), torch.zeros_like(theta), torch.sin(theta),
                torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta),
                -torch.sin(theta), torch.zeros_like(theta), torch.cos(theta),
            ], dim=-1).reshape(bsz, 3, 3)
            inv_rot = rot.transpose(1, 2)
        else:
            rot = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(bsz, 1, 1)
            inv_rot = rot.transpose(1, 2)

        x = torch.einsum('bij,bjk->bik', x, rot)
        x_his = torch.einsum('bcij,bjk->bcik', x_his, rot)
        v = torch.einsum('bij,bjk->bik', v, rot)
        v_his = torch.einsum('bcij,bjk->bcik', v_his, rot)

        x_grid = x_grid - x_center
        x_grid = x_grid @ rot
        x_his = x_his.reshape(bsz, num_particles, self.n_history * 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history * 3)

        # Build per-particle feature vector
        geom_feat = torch.cat([x, v, x_his, v_his], dim=-1)  # (B, N, 6*(1+n_history))

        # Choose encoder based on whether visual features are available
        if vis_feat is not None and self.encoder_visual is not None:
            # Concatenate visual features with geometric features
            combined_feat = torch.cat([geom_feat, vis_feat], dim=-1)  # (B, N, geom_ch + vis_dim)
            combined_feat = combined_feat.permute(0, 2, 1)  # (B, C, N)
            feat, trans, trans_feat = self.encoder_visual(combined_feat, enabled)
        else:
            # Geometry-only path (identical to original PGNDModel)
            geom_feat = geom_feat.permute(0, 2, 1)  # (B, C, N)
            feat, trans, trans_feat = self.encoder(geom_feat, enabled)

        # Grid aggregation
        if self.radius > 0:
            dist_pt_grid = torch.cdist(x_grid, x, p=2)
            mask = dist_pt_grid < self.radius
            mask_normed = mask / (mask.sum(dim=-1, keepdim=True) + 1e-5)
            mask_normed = mask_normed.detach()
            feat = mask_normed @ feat.permute(0, 2, 1)
        else:
            feat = feat[:, None, :].repeat(1, x_grid.shape[1], 1)

        # Positional encoding + decoder
        feat = feat.reshape(-1, self.feature_dim)
        x_grid = x_grid.reshape(-1, 3)
        x_grid = self.positional_encoding(x_grid)
        feat = self.decoder(x_grid, feat)
        feat = feat * self.output_scale
        feat = feat.reshape(bsz, -1, feat.shape[-1])
        feat = torch.bmm(feat, inv_rot)

        # Expand to full grid
        feat = fill_grid_locations(feat, grid_idxs, self.num_grids_list)
        return feat
