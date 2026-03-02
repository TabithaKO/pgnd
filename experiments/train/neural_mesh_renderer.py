"""
neural_mesh_renderer.py — Neural Mesh-Conditioned Gaussian Renderer
=====================================================================

A learned renderer that takes ANY mesh (vertices + faces) and camera params,
and produces an RGB rendering. No per-scene optimization needed at inference.

Key idea: Instead of storing per-face GS params (which are topology-dependent),
we predict GS params from LOCAL geometric features of each face. This makes
the renderer topology-agnostic and generalizable across particle configurations.

Architecture:
    Input mesh (vertices, faces) + vertex colors
            ↓
    Per-face geometric feature extraction:
        - face normal, area, edge lengths
        - vertex colors (from point cloud projection)
        - relative vertex positions (local frame)
            ↓
    Feature MLP: per-face features → per-Gaussian params
        - colors (3), scales (3), opacity (1), bary offsets (6 for 2 per face)
            ↓
    Barycentric interpolation → Gaussian 3D positions
            ↓
    Differentiable GS rasterization → RGB image

Usage:
    renderer = NeuralMeshRenderer(feature_dim=32, hidden_dim=128)

    # Training: mesh built from current particles
    mesh = build_mesh(particles)
    vertex_colors = project_colors(particles, camera, image)
    rendered = renderer(mesh, vertex_colors, camera_settings)
    loss = image_loss(rendered, gt_image)
    loss.backward()  # gradients flow to MLP params AND particle positions

    # Inference: any mesh + camera → RGB
    rendered = renderer(new_mesh, new_vertex_colors, new_camera)
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import GaussianRasterizer
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    HAS_GS_RASTERIZER = True
except ImportError:
    HAS_GS_RASTERIZER = False
    print("[neural_mesh_renderer] Warning: diff_gaussian_rasterization not available")

try:
    import roma
    HAS_ROMA = True
except ImportError:
    HAS_ROMA = False

try:
    from build_cloth_mesh import compute_mesh_from_particles
except ImportError:
    print("[neural_mesh_renderer] Warning: Could not import mesh utilities")


# =============================================================================
# Activations (matching 3DGS conventions)
# =============================================================================

def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-8) + 1e-8)


# =============================================================================
# Per-Face Feature Extraction
# =============================================================================

class MeshFeatureExtractor:
    """Extract local geometric features from mesh faces.

    For each face, computes features that are invariant to global mesh topology
    but capture local surface properties. These features are what the MLP
    uses to predict Gaussian appearance.
    """

    @staticmethod
    def compute_face_features(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: Optional[torch.Tensor] = None,
        view_dir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-face geometric features.

        Args:
            vertices: (N_verts, 3) vertex positions
            faces: (3, N_faces) face vertex indices
            vertex_colors: (N_verts, 3) optional RGB colors per vertex
            view_dir: (3,) camera-to-scene direction in same space as vertices,
                      or None to skip view-dependent features

        Returns:
            (N_faces, feature_dim) per-face feature vectors
            Feature composition:
                - face normal (3)
                - face area (1)
                - edge lengths (3)
                - relative vertex positions (9 = 3 vertices * 3)
                - view direction (3) [if provided]
                - normal·view dot product (1) [if provided]
                - per-face view direction from centroid (3) [if provided]
                - vertex colors mean (3) [if provided]
                - vertex colors per-vertex (9 = 3 vertices * 3) [if provided]
            Total: 16 base + 7 view + 12 color = 35 max
        """
        n_faces = faces.shape[1]

        # Get face vertex positions: (N_faces, 3_verts, 3_xyz)
        v0 = vertices[faces[0]]  # (N_faces, 3)
        v1 = vertices[faces[1]]
        v2 = vertices[faces[2]]

        # Face centroid
        centroid = (v0 + v1 + v2) / 3.0  # (N_faces, 3)

        # Edges
        e01 = v1 - v0
        e02 = v2 - v0
        e12 = v2 - v1

        # Face normal (unnormalized → contains area info)
        cross = torch.cross(e01, e02, dim=1)  # (N_faces, 3)
        area = torch.norm(cross, dim=1, keepdim=True) * 0.5  # (N_faces, 1)
        normal = F.normalize(cross, dim=1)  # (N_faces, 3)

        # Edge lengths
        len01 = torch.norm(e01, dim=1, keepdim=True)  # (N_faces, 1)
        len02 = torch.norm(e02, dim=1, keepdim=True)
        len12 = torch.norm(e12, dim=1, keepdim=True)

        # Relative vertex positions (in local frame centered at centroid)
        rv0 = v0 - centroid  # (N_faces, 3)
        rv1 = v1 - centroid
        rv2 = v2 - centroid

        features = [
            normal,           # (N_faces, 3) - surface orientation
            area,             # (N_faces, 1) - face size
            len01, len02, len12,  # (N_faces, 3) - edge lengths
            rv0, rv1, rv2,    # (N_faces, 9) - local geometry
        ]

        # View-dependent features
        if view_dir is not None:
            # Global view direction (same for all faces)
            view_dir_norm = F.normalize(view_dir, dim=0)
            view_broadcast = view_dir_norm.unsqueeze(0).expand(n_faces, -1)  # (N_faces, 3)

            # Normal · view = how much the face points toward camera
            # Positive = facing camera, negative = facing away
            ndotv = (normal * view_broadcast).sum(dim=1, keepdim=True)  # (N_faces, 1)

            # Per-face view direction from centroid to camera (local parallax)
            # camera_pos is implicit: view_dir points from camera to scene center
            # So per-face direction = -view_dir relative to centroid (approximation)
            # For more accuracy, compute per-face: cam_pos - centroid
            per_face_view = F.normalize(-view_broadcast - centroid + centroid.mean(dim=0, keepdim=True), dim=1)

            features.extend([
                view_broadcast,   # (N_faces, 3) - global view direction
                ndotv,            # (N_faces, 1) - facing ratio
                per_face_view,    # (N_faces, 3) - per-face view direction
            ])

        if vertex_colors is not None:
            c0 = vertex_colors[faces[0]]  # (N_faces, 3)
            c1 = vertex_colors[faces[1]]
            c2 = vertex_colors[faces[2]]
            mean_color = (c0 + c1 + c2) / 3.0

            features.extend([
                mean_color,   # (N_faces, 3) - average face color
                c0, c1, c2,  # (N_faces, 9) - per-vertex colors
            ])

        return torch.cat(features, dim=1)  # (N_faces, feat_dim)


# =============================================================================
# Neural Mesh Renderer
# =============================================================================

class NeuralMeshRenderer(nn.Module):
    """Neural renderer: mesh + vertex colors + camera → RGB image.

    Learns to predict per-Gaussian appearance (color, scale, opacity) from
    local mesh features. Topology-agnostic: works for any mesh configuration.

    The MLP predicts per-face Gaussian parameters:
        - colors: (n_gauss_per_face, 3) RGB
        - scales: (n_gauss_per_face, 3) log-space
        - opacities: (n_gauss_per_face, 1) logit-space
        - bary_offsets: (n_gauss_per_face, 3) barycentric coordinate offsets from center

    Architecture:
        face_features → MLP → per-Gaussian params → rasterize → image
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        gaussians_per_face: int = 4,
        feature_dim_no_color: int = 16,
        feature_dim_with_color: int = 28,
        feature_dim_view: int = 7,  # view_dir(3) + ndotv(1) + per_face_view(3)
        use_vertex_colors: bool = True,
        use_view_direction: bool = True,
    ):
        super().__init__()

        self.gaussians_per_face = gaussians_per_face
        self.use_vertex_colors = use_vertex_colors
        self.use_view_direction = use_view_direction
        self.feature_extractor = MeshFeatureExtractor()

        input_dim = feature_dim_no_color
        if use_vertex_colors:
            input_dim = feature_dim_with_color
        if use_view_direction:
            input_dim += feature_dim_view

        # Per-Gaussian output dimensions
        # For each Gaussian: color(3) + scale(3) + opacity(1) + bary(3) = 10
        output_per_gaussian = 10
        output_dim = gaussians_per_face * output_per_gaussian

        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize output layer to produce reasonable defaults
        self._init_output_layer()

        # LPIPS perceptual loss (lazy-loaded)
        self._lpips_net = None

    def _init_output_layer(self):
        """Initialize the final layer so initial predictions are reasonable.

        Goal: colors near 0.5 (gray), scales near log(0.005), opacity near 0.5,
        bary coords near (1/3, 1/3, 1/3).
        """
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)

        with torch.no_grad():
            bias = final_layer.bias
            n_gpf = self.gaussians_per_face

            for g in range(n_gpf):
                offset = g * 10
                # Colors: sigmoid(0) = 0.5 (gray)
                bias[offset:offset + 3] = 0.0
                # Scales: exp(-5.3) ≈ 0.005
                bias[offset + 3:offset + 6] = -5.3
                # Opacity: sigmoid(0) = 0.5
                bias[offset + 6] = 0.0
                # Bary: softmax([0,0,0]) = (1/3, 1/3, 1/3)
                bias[offset + 7:offset + 10] = 0.0

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: Optional[torch.Tensor],
        cam_settings: Dict,
        coord_transform=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Render mesh to RGB image.

        Args:
            vertices: (N_verts, 3) vertex positions (PGND preprocessed space)
            faces: (3, N_faces) face vertex indices
            vertex_colors: (N_verts, 3) RGB colors per vertex [0,1], or None
            cam_settings: dict with 'w', 'h', 'k' (intrinsics), 'w2c' (world-to-camera)
            coord_transform: PGNDCoordinateTransform for preprocessed → world space

        Returns:
            rendered_image: (3, H, W) RGB image
            debug_info: dict with intermediate values for debugging
        """
        n_faces = faces.shape[1]

        # =====================================================================
        # Step 1: Compute view direction from camera
        # =====================================================================
        view_dir = None
        if self.use_view_direction and cam_settings is not None:
            w2c = cam_settings['w2c']
            # Camera position in world space: c = -R^T @ t
            R_cam = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=vertices.device)
            t_cam = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=vertices.device)
            cam_pos_world = -R_cam.T @ t_cam  # (3,)

            # If we have coord_transform, transform camera to preprocessed space
            # (since vertices are in preprocessed space for feature extraction)
            if coord_transform is not None:
                cam_pos_preproc = coord_transform.forward_transform(cam_pos_world.unsqueeze(0)).squeeze(0)
            else:
                cam_pos_preproc = cam_pos_world

            # View direction: from camera to scene center (mean of vertices)
            scene_center = vertices.detach().mean(dim=0)
            view_dir = F.normalize(scene_center - cam_pos_preproc, dim=0)

        # =====================================================================
        # Step 2: Extract per-face geometric features
        # =====================================================================
        face_features = self.feature_extractor.compute_face_features(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors if self.use_vertex_colors else None,
            view_dir=view_dir,
        )  # (N_faces, feat_dim)

        # =====================================================================
        # Step 2: MLP predicts per-Gaussian parameters
        # =====================================================================
        raw_output = self.mlp(face_features)  # (N_faces, gaussians_per_face * 10)

        # Reshape to (N_faces, gaussians_per_face, 10)
        raw_output = raw_output.view(n_faces, self.gaussians_per_face, 10)

        # Parse into per-Gaussian params
        raw_colors = raw_output[:, :, 0:3]     # (N_faces, gpf, 3)
        raw_scales = raw_output[:, :, 3:6]     # (N_faces, gpf, 3)
        raw_opacities = raw_output[:, :, 6:7]  # (N_faces, gpf, 1)
        raw_bary = raw_output[:, :, 7:10]      # (N_faces, gpf, 3)

        # Apply activations
        colors = torch.sigmoid(raw_colors)                    # [0, 1]
        scales = torch.exp(raw_scales)                        # positive
        opacities = torch.sigmoid(raw_opacities)              # [0, 1]
        bary_coords = F.softmax(raw_bary, dim=-1)            # sum to 1, all positive

        # Flatten from (N_faces, gpf, ...) to (N_gaussians, ...)
        n_gaussians = n_faces * self.gaussians_per_face
        colors = colors.reshape(n_gaussians, 3)
        scales = scales.reshape(n_gaussians, 3)
        opacities = opacities.reshape(n_gaussians, 1)
        bary_coords = bary_coords.reshape(n_gaussians, 3)

        # =====================================================================
        # Step 3: Compute Gaussian positions via barycentric interpolation
        # =====================================================================

        # Face IDs for each Gaussian: [0,0, 1,1, 2,2, ...] for gpf=2
        face_ids = torch.arange(
            n_faces, dtype=torch.long, device=vertices.device
        ).repeat_interleave(self.gaussians_per_face)

        # Get face vertex positions
        vertice_ids = faces[:, face_ids].transpose(0, 1)  # (N_gaussians, 3)
        face_pos = vertices[vertice_ids, :]  # (N_gaussians, 3_verts, 3_xyz)

        # Barycentric interpolation: pos = bary @ face_vertices
        gaussian_positions = (bary_coords.unsqueeze(1) @ face_pos).squeeze(1)  # (N_gaussians, 3)

        # =====================================================================
        # Step 4: Compute Gaussian rotations from face deformation
        # =====================================================================

        # Identity quaternion as default
        rotations = torch.zeros(n_gaussians, 4, device=vertices.device)
        rotations[:, 0] = 1.0  # w=1, x=y=z=0

        # TODO: could add roma rigid registration for deformed faces
        # For now, identity is fine since we're predicting appearance, not tracking

        # =====================================================================
        # Step 5: Transform to world space and render
        # =====================================================================

        if coord_transform is not None:
            gaussian_positions_world = coord_transform.inverse_transform(gaussian_positions)
        else:
            gaussian_positions_world = gaussian_positions

        # Setup camera
        from render_loss import setup_camera_for_render
        cam = setup_camera_for_render(
            w=cam_settings['w'],
            h=cam_settings['h'],
            k=cam_settings['k'],
            w2c=cam_settings['w2c'],
        )

        render_data = {
            'means3D': gaussian_positions_world,
            'colors_precomp': colors,
            'rotations': F.normalize(rotations, dim=-1),
            'opacities': opacities,
            'scales': scales,
            'means2D': torch.zeros_like(gaussian_positions_world, requires_grad=True, device=vertices.device) + 0,
        }

        rendered_image, _, _ = GaussianRasterizer(raster_settings=cam)(**render_data)

        debug_info = {
            'n_gaussians': n_gaussians,
            'n_faces': n_faces,
            'gaussian_positions': gaussian_positions,
            'colors': colors,
            'scales': scales,
            'opacities': opacities,
            'bary_coords': bary_coords,
        }

        return rendered_image, debug_info

    def compute_lpips_loss(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute LPIPS perceptual loss between rendered and GT images.

        Args:
            rendered: (3, H, W) rendered image [0, 1]
            gt: (3, H, W) ground truth image [0, 1]
            mask: (1, H, W) optional mask

        Returns:
            Scalar LPIPS loss
        """
        if self._lpips_net is None:
            try:
                import lpips
                self._lpips_net = lpips.LPIPS(net='vgg').to(rendered.device)
                self._lpips_net.eval()
                for p in self._lpips_net.parameters():
                    p.requires_grad = False
                print("[NeuralMeshRenderer] LPIPS (VGG) loss initialized")
            except ImportError:
                print("[NeuralMeshRenderer] WARNING: lpips not installed, skipping perceptual loss")
                return torch.tensor(0.0, device=rendered.device)

        # LPIPS expects (B, 3, H, W) in [-1, 1]
        rendered_lpips = rendered.unsqueeze(0) * 2.0 - 1.0
        gt_lpips = gt.unsqueeze(0) * 2.0 - 1.0

        if mask is not None:
            rendered_lpips = rendered_lpips * mask.unsqueeze(0)
            gt_lpips = gt_lpips * mask.unsqueeze(0)

        return self._lpips_net(rendered_lpips, gt_lpips).mean()

    def render_with_particles(
        self,
        particles: torch.Tensor,
        cam_settings: Dict,
        coord_transform,
        vertex_colors: Optional[torch.Tensor] = None,
        mesh_method: str = 'bpa',
    ) -> Tuple[torch.Tensor, Dict]:
        """Convenience method: particles → mesh → render.

        Builds mesh from particles, extracts vertex colors if not provided,
        and renders.

        Args:
            particles: (N, 3) particle positions in PGND preprocessed space
            cam_settings: camera settings dict
            coord_transform: PGNDCoordinateTransform
            vertex_colors: (N, 3) optional pre-computed vertex colors
            mesh_method: mesh construction method

        Returns:
            rendered_image: (3, H, W) RGB
            debug_info: dict
        """
        # Build mesh from particles
        mesh_data = compute_mesh_from_particles(particles.detach(), method=mesh_method)
        vertices = mesh_data.pos  # (N_used_verts, 3) — may be subset of particles
        faces = mesh_data.face    # (3, N_faces)

        # If vertex colors provided for all particles, index into used vertices
        if vertex_colors is not None and vertex_colors.shape[0] != vertices.shape[0]:
            # mesh may use a subset of particles
            # For now, truncate or pad
            n_verts = vertices.shape[0]
            if vertex_colors.shape[0] >= n_verts:
                vertex_colors = vertex_colors[:n_verts]
            else:
                # Pad with gray
                pad = torch.full(
                    (n_verts - vertex_colors.shape[0], 3),
                    0.5, device=vertex_colors.device
                )
                vertex_colors = torch.cat([vertex_colors, pad], dim=0)

        return self.forward(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            cam_settings=cam_settings,
            coord_transform=coord_transform,
        )


# =============================================================================
# Vertex Color Utilities
# =============================================================================

def project_vertex_colors(
    vertices_preproc: torch.Tensor,
    image: torch.Tensor,
    cam_settings: Dict,
    coord_transform,
) -> torch.Tensor:
    """Project vertices onto camera image to get per-vertex RGB colors.

    Args:
        vertices_preproc: (N, 3) vertex positions in PGND preprocessed space
        image: (3, H, W) GT image [0, 1]
        cam_settings: dict with 'k', 'w2c'
        coord_transform: PGNDCoordinateTransform

    Returns:
        (N, 3) RGB colors per vertex [0, 1]
    """
    with torch.no_grad():
        # Transform to world space
        vertices_world = coord_transform.inverse_transform(vertices_preproc)
        pts_np = vertices_world.cpu().numpy()

        K = cam_settings['k']
        w2c = cam_settings['w2c']
        R_cam = w2c[:3, :3]
        t_cam = w2c[:3, 3]

        # Project to camera
        pts_cam = (R_cam @ pts_np.T + t_cam.reshape(3, 1)).T
        pts_2d = (K @ pts_cam.T).T
        u = pts_2d[:, 0] / (pts_2d[:, 2] + 1e-8)
        v = pts_2d[:, 1] / (pts_2d[:, 2] + 1e-8)

        H, W = image.shape[1], image.shape[2]
        u_px = torch.from_numpy(u).long().clamp(0, W - 1).to(image.device)
        v_px = torch.from_numpy(v).long().clamp(0, H - 1).to(image.device)

        # Sample colors
        colors = image[:, v_px, u_px].T  # (N, 3)

        # Mark valid projections
        valid = (
            (pts_cam[:, 2] > 0) &
            (u >= 0) & (u < W) &
            (v >= 0) & (v < H)
        )
        valid = torch.from_numpy(valid).to(image.device)

        if valid.sum() > 0:
            mean_color = colors[valid].mean(dim=0)
            colors[~valid] = mean_color
        else:
            colors[:] = 0.5

        return colors


# =============================================================================
# Factory
# =============================================================================

def create_neural_mesh_renderer(
    hidden_dim: int = 256,
    n_hidden_layers: int = 4,
    gaussians_per_face: int = 4,
    use_vertex_colors: bool = True,
    use_view_direction: bool = True,
    device: str = 'cuda',
) -> NeuralMeshRenderer:
    """Create and initialize a NeuralMeshRenderer."""
    renderer = NeuralMeshRenderer(
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        gaussians_per_face=gaussians_per_face,
        use_vertex_colors=use_vertex_colors,
        use_view_direction=use_view_direction,
    )
    renderer = renderer.to(device)
    print(f"[NeuralMeshRenderer] Created: hidden={hidden_dim}, layers={n_hidden_layers}, "
          f"gpf={gaussians_per_face}, view_dep={use_view_direction}, "
          f"params={sum(p.numel() for p in renderer.parameters()):,}")
    return renderer