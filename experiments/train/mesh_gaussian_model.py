"""
mesh_gaussian_model.py — Mesh-Constrained Gaussian Model (Cloth-Splatting Style)
==================================================================================

Follows cloth-splatting's MultiGaussianMesh approach:
  - Gaussians are CREATED on mesh faces (not anchored from a pre-trained splat)
  - Each face gets `gaussian_init_factor` Gaussians with random barycentric coords
  - All GS parameters (color, scale, rotation, opacity) are randomly initialized
  - The render loss trains everything from scratch

Architecture:
    PGND dynamics model (fθ) predicts particle positions
            ↓
    Particles = mesh vertices (1:1 mapping, BPA/Delaunay mesh)
            ↓
    Gaussian positions = barycentric interpolation on mesh faces
            ↓
    Differentiable GS rasterization
            ↓
    Loss: image loss trains both dynamics model AND GS parameters

Reference: cloth-splatting/scene_reconstruction/gaussian_mesh.py
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data
import torch_geometric.transforms

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from build_cloth_mesh import load_pgnd_particles, preprocess_particles, compute_mesh_from_particles
except ImportError:
    print("[mesh_gaussian_model] Warning: Could not import mesh utilities")

try:
    import roma
    HAS_ROMA = True
except ImportError:
    HAS_ROMA = False


# =============================================================================
# Activations (matching cloth-splatting / 3DGS conventions)
# =============================================================================

def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-8) + 1e-8)


def sigmoid_activation(x):
    return torch.sigmoid(x)


def scaling_activation(x):
    return torch.exp(x)


def rotation_activation(x):
    return torch.nn.functional.normalize(x, dim=-1)


# =============================================================================
# Mesh-Constrained Gaussian Model (Cloth-Splatting Style)
# =============================================================================

class MeshGaussianModel(nn.Module):
    """Mesh-constrained Gaussian Splatting model following cloth-splatting.

    Gaussians live ON mesh faces via barycentric coordinates.
    All parameters are randomly initialized and learned via render loss.

    Trainable parameters:
        - face_bary: (n_gauss, 3) barycentric coordinates on assigned face
        - _colors: (n_gauss, 1, 3) DC color coefficients (SH degree 0)
        - _scales: (n_gauss, 3) log-space scales
        - _rotations: (n_gauss, 4) quaternions
        - _opacities: (n_gauss, 1) logit-space opacities

    Fixed:
        - mesh: topology (pos, face, edge_index)
        - face_ids: (n_gauss,) which face each Gaussian belongs to
    """

    def __init__(self, sh_degree: int = 0):
        super().__init__()

        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        # Mesh topology (fixed per episode)
        self.mesh = torch_geometric.data.Data()

        # Gaussian-to-face assignment (fixed after init)
        self.register_buffer('face_ids', torch.empty(0, dtype=torch.long))

        # Trainable parameters
        self.face_bary = nn.Parameter(torch.empty(0, 3))
        self._colors = nn.Parameter(torch.empty(0, 1, 3))  # SH DC term
        self._scales = nn.Parameter(torch.empty(0, 3))
        self._rotations = nn.Parameter(torch.empty(0, 4))
        self._opacities = nn.Parameter(torch.empty(0, 1))

        self.spatial_lr_scale = 1.0

    @property
    def num_gaussians(self) -> int:
        return self.face_ids.shape[0]

    # =========================================================================
    # Initialization: create Gaussians ON mesh faces
    # =========================================================================

    def initialize_from_mesh(
        self,
        particles: torch.Tensor,
        method: str = 'bpa',
        gaussian_init_factor: int = 2,
        spatial_lr_scale: float = 1.0,
        device: str = 'cuda',
    ):
        """Initialize model by building mesh from particles and placing Gaussians on faces.

        Follows cloth-splatting's _setup_callback:
        1. Build mesh from particles (Delaunay triangulation)
        2. Assign `gaussian_init_factor` Gaussians per face
        3. Random barycentric coords near face center
        4. Random colors, identity rotations, KNN-based scales, low opacity

        Args:
            particles: (N, 3) particle positions (already in preprocessed space)
            method: mesh construction method ('bpa' = Delaunay)
            gaussian_init_factor: number of Gaussians per face (cloth-splatting uses 2)
            spatial_lr_scale: learning rate scale for position params
            device: torch device
        """
        self.spatial_lr_scale = spatial_lr_scale

        # =====================================================================
        # Step 1: Build mesh from particles
        # =====================================================================
        mesh_data = compute_mesh_from_particles(
            particles, method=method,
        )
        self.mesh = mesh_data
        n_faces = self.mesh.face.shape[1]
        n_vertices = self.mesh.pos.shape[0]

        print(f"[MeshGaussianModel] Mesh: {n_vertices} vertices, {n_faces} faces")

        # =====================================================================
        # Step 2: Assign Gaussians to faces
        # =====================================================================
        n_gaussians = gaussian_init_factor * n_faces
        print(f"[MeshGaussianModel] Gaussians: {n_gaussians} ({gaussian_init_factor} per face)")

        # Each face gets gaussian_init_factor Gaussians
        # face_ids: [0,0, 1,1, 2,2, ...] for factor=2
        self.face_ids = torch.arange(
            0, n_faces, dtype=torch.long, device=device
        ).repeat(gaussian_init_factor).sort().values

        # =====================================================================
        # Step 3: Random barycentric coordinates near face center
        # =====================================================================
        # Cloth-splatting: start at (1/3, 1/3, 1/3) with small noise
        face_bary = torch.ones((n_gaussians, 3), dtype=torch.float, device=device) / 3.0
        if gaussian_init_factor > 1:
            face_bary = torch.clip(torch.normal(face_bary, 0.05), 0.0, 1.0)
            face_bary = face_bary / face_bary.sum(dim=1, keepdim=True)
        self.face_bary = nn.Parameter(face_bary.requires_grad_(True))

        # =====================================================================
        # Step 4: Initialize GS parameters (all random, learned from scratch)
        # =====================================================================

        # Colors: random small values (cloth-splatting uses random/255)
        # Init colors to mid-gray so Gaussians are visible immediately
        colors_np = (np.random.random((n_gaussians, 3)).astype(np.float32) * 0.3 + 0.35)
        colors = torch.from_numpy(colors_np).float().to(device)
        # Store as (n_gauss, 1, 3) for SH DC term compatibility
        self._colors = nn.Parameter(
            colors.unsqueeze(1).contiguous().requires_grad_(True)
        )

        # Scales: based on KNN distance between Gaussian positions
        # Cloth-splatting uses distCUDA2; we use a simple KNN fallback
        with torch.no_grad():
            point_coords = self.get_xyz()  # compute initial positions from bary
            dist2 = self._compute_knn_dist2(point_coords)
            # Scale up by 3x so Gaussians overlap and form a solid surface
            scales = torch.log(torch.sqrt(dist2) * 1.5).unsqueeze(-1).repeat(1, 3)
        self._scales = nn.Parameter(scales.requires_grad_(True))

        # Rotations: identity quaternion (w=1, x=y=z=0)
        rots = torch.zeros((n_gaussians, 4), device=device)
        rots[:, 0] = 1
        self._rotations = nn.Parameter(rots.requires_grad_(True))

        # Opacities: low initial opacity (sigmoid^-1(0.1))
        opacities = inverse_sigmoid(
            0.5 * torch.ones((n_gaussians, 1), dtype=torch.float, device=device)
        )
        self._opacities = nn.Parameter(opacities.requires_grad_(True))

        print(f"[MeshGaussianModel] Initialization complete!")
        print(f"  Trainable: face_bary, colors, scales, rotations, opacities")

    def initialize_colors_from_image(
        self,
        vertices_world: torch.Tensor,
        image: torch.Tensor,
        intrinsics: np.ndarray,
        w2c: np.ndarray,
    ):
        """Initialize Gaussian colors by projecting onto a camera image.

        For each Gaussian, compute its world-space position from face vertices,
        project to 2D, and sample the pixel color.

        Args:
            vertices_world: (N_vertices, 3) mesh vertices in world space
            image: (3, H, W) GT image tensor [0,1]
            intrinsics: (3, 3) camera intrinsic matrix
            w2c: (4, 4) world-to-camera transform
        """
        with torch.no_grad():
            # Get Gaussian positions in world space
            gs_positions = self.get_xyz(deformed_vertices=vertices_world)  # (N_gauss, 3)

            # Project to camera space
            w2c_t = torch.from_numpy(w2c).float().to(gs_positions.device)
            K = torch.from_numpy(intrinsics).float().to(gs_positions.device)

            # Homogeneous coords
            ones = torch.ones(gs_positions.shape[0], 1, device=gs_positions.device)
            pts_h = torch.cat([gs_positions, ones], dim=1)  # (N, 4)

            # World to camera
            pts_cam = (w2c_t @ pts_h.T).T[:, :3]  # (N, 3)

            # Project to 2D
            pts_2d = (K @ pts_cam.T).T  # (N, 3)
            u = pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)
            v = pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)

            H, W = image.shape[1], image.shape[2]

            # Clamp to image bounds
            u_px = u.long().clamp(0, W - 1)
            v_px = v.long().clamp(0, H - 1)

            # Sample colors
            colors = image[:, v_px, u_px].T  # (N_gauss, 3)

            # Check which projections are valid (in front of camera, within image)
            valid = (pts_cam[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            # For invalid projections, use mean color of valid ones
            if valid.sum() > 0:
                mean_color = colors[valid].mean(dim=0)
                colors[~valid] = mean_color

            # Update color parameter
            self._colors.data[:, 0, :] = colors
            
            n_valid = valid.sum().item()
            print(f"  [color init] {n_valid}/{len(valid)} Gaussians projected onto image")
            print(f"  [color init] Color range: [{colors.min():.3f}, {colors.max():.3f}]")

    def _compute_knn_dist2(self, points: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Compute squared distance to k-th nearest neighbor for each point.

        Fallback for distCUDA2 which requires simple_knn.

        Args:
            points: (N, 3) point positions
            k: number of neighbors

        Returns:
            (N,) squared distances to k-th nearest neighbor
        """
        from sklearn.neighbors import NearestNeighbors
        pts_np = points.detach().cpu().numpy()
        k_actual = min(k + 1, len(pts_np))  # +1 because self is included
        knn = NearestNeighbors(n_neighbors=k_actual).fit(pts_np)
        distances, _ = knn.kneighbors(pts_np)
        # Use mean of k nearest (excluding self at index 0)
        mean_dist = distances[:, 1:].mean(axis=1)
        dist2 = torch.from_numpy(mean_dist ** 2).float().to(points.device)
        return torch.clamp_min(dist2, 1e-7)

    # =========================================================================
    # Gaussian position/rotation from mesh deformation
    # =========================================================================

    def get_xyz(self, deformed_vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Gaussian positions via barycentric interpolation on mesh faces.

        Args:
            deformed_vertices: (N_vertices, 3) deformed mesh positions.
                If None, use original mesh.pos.

        Returns:
            (N_gaussians, 3) Gaussian positions
        """
        vertice_ids = self.mesh.face[:, self.face_ids].transpose(0, 1)  # (N_gauss, 3)

        if deformed_vertices is not None:
            assert deformed_vertices.shape[0] == self.mesh.pos.shape[0], \
                f"Vertex count mismatch: expected {self.mesh.pos.shape[0]}, got {deformed_vertices.shape[0]}"
            face_pos = deformed_vertices[vertice_ids, :]
        else:
            face_pos = self.mesh.pos[vertice_ids, :]  # (N_gauss, 3, 3)

        norm_bary = self.face_bary / (self.face_bary.sum(dim=1, keepdim=True) + 1e-8)
        pos = (norm_bary.unsqueeze(1) @ face_pos).squeeze(1)
        return pos

    def get_rotation(self, deformed_vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Gaussian rotations, optionally with deformation-induced rotation.

        Follows cloth-splatting: base rotation composed with rigid registration
        of face vertices (original → deformed).

        Args:
            deformed_vertices: (N_vertices, 3) optional deformed mesh positions

        Returns:
            (N_gaussians, 4) quaternions
        """
        rotation = rotation_activation(self._rotations)

        if deformed_vertices is None or not HAS_ROMA:
            return rotation

        vertice_ids = self.mesh.face[:, self.face_ids].transpose(0, 1)
        vertice_pos = self.mesh.pos[vertice_ids, :]
        deformed_vertice_pos = deformed_vertices[vertice_ids, :]

        try:
            relative_rotation, _ = roma.rigid_points_registration(
                vertice_pos, deformed_vertice_pos
            )
            relative_rotation_quat = roma.rotmat_to_unitquat(relative_rotation)
            return roma.quat_composition([rotation, relative_rotation_quat])
        except Exception as e:
            return rotation

    # =========================================================================
    # Parameter getters with activations
    # =========================================================================

    def get_colors(self) -> torch.Tensor:
        """Get RGB colors from SH DC coefficients.

        Returns:
            (N_gaussians, 3) RGB colors
        """
        # _colors is (N, 1, 3) SH DC term
        # For sh_degree=0, just squeeze and clamp
        return torch.clamp(self._colors.squeeze(1), 0.0, 1.0)

    def get_scales(self) -> torch.Tensor:
        return scaling_activation(self._scales)

    def get_opacities(self) -> torch.Tensor:
        return sigmoid_activation(self._opacities)

    # =========================================================================
    # Optimizer setup
    # =========================================================================

    def get_optimizer_param_groups(
        self,
        lr_position: float = 1e-4,
        lr_color: float = 2.5e-3,
        lr_scale: float = 5e-3,
        lr_rotation: float = 1e-3,
        lr_opacity: float = 5e-2,
    ) -> list:
        """Parameter groups with per-parameter learning rates (matching cloth-splatting)."""
        return [
            {'params': [self.face_bary], 'lr': lr_position * self.spatial_lr_scale, 'name': 'face_bary'},
            {'params': [self._colors], 'lr': lr_color, 'name': 'colors'},
            {'params': [self._scales], 'lr': lr_scale, 'name': 'scales'},
            {'params': [self._rotations], 'lr': lr_rotation, 'name': 'rotations'},
            {'params': [self._opacities], 'lr': lr_opacity, 'name': 'opacities'},
        ]

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(save_dir / 'mesh.npz'),
            pos=self.mesh.pos.cpu().numpy(),
            face=self.mesh.face.cpu().numpy(),
            edge_index=self.mesh.edge_index.cpu().numpy() if hasattr(self.mesh, 'edge_index') and self.mesh.edge_index is not None else np.array([]),
        )

        np.savez_compressed(
            str(save_dir / 'gaussian_params.npz'),
            face_bary=self.face_bary.detach().cpu().numpy(),
            face_ids=self.face_ids.cpu().numpy(),
            colors=self._colors.detach().cpu().numpy(),
            scales=self._scales.detach().cpu().numpy(),
            rotations=self._rotations.detach().cpu().numpy(),
            opacities=self._opacities.detach().cpu().numpy(),
        )

        print(f"[MeshGaussianModel] Saved to {save_dir}")

    def load(self, load_dir: Path, device='cuda'):
        mesh_data = np.load(str(load_dir / 'mesh.npz'))
        self.mesh.pos = torch.from_numpy(mesh_data['pos']).float().to(device)
        self.mesh.face = torch.from_numpy(mesh_data['face']).long().to(device)
        if mesh_data['edge_index'].size > 0:
            self.mesh.edge_index = torch.from_numpy(mesh_data['edge_index']).long().to(device)

        gs_data = np.load(str(load_dir / 'gaussian_params.npz'))
        self.face_ids = torch.from_numpy(gs_data['face_ids']).long().to(device)
        self.face_bary = nn.Parameter(
            torch.from_numpy(gs_data['face_bary']).float().to(device).requires_grad_(True)
        )
        self._colors = nn.Parameter(
            torch.from_numpy(gs_data['colors']).float().to(device).requires_grad_(True)
        )
        self._scales = nn.Parameter(
            torch.from_numpy(gs_data['scales']).float().to(device).requires_grad_(True)
        )
        self._rotations = nn.Parameter(
            torch.from_numpy(gs_data['rotations']).float().to(device).requires_grad_(True)
        )
        self._opacities = nn.Parameter(
            torch.from_numpy(gs_data['opacities']).float().to(device).requires_grad_(True)
        )

        print(f"[MeshGaussianModel] Loaded from {load_dir}")
        print(f"  {self.num_gaussians} Gaussians, {self.mesh.pos.shape[0]} vertices, {self.mesh.face.shape[1]} faces")