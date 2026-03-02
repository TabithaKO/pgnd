"""
build_cloth_mesh.py — Mesh Construction for Cloth-Constrained Gaussian Splatting
==================================================================================

Builds a triangular mesh from PGND particle point clouds using Delaunay
triangulation. The mesh topology is fixed per episode and used to constrain
Gaussian Splatting parameters via barycentric coordinates.

Usage:
    python build_cloth_mesh.py --episode_dir /path/to/episode_XXXX

Output:
    - mesh.npz: Contains vertices (first frame), faces, edge_index
    - mesh_viz.ply: PLY file for visualization in MeshLab/CloudCompare

Dependencies:
    - scipy (for Delaunay triangulation)
    - torch_geometric (for mesh data structure)
    - open3d (optional, for visualization)
"""

import argparse
import numpy as np
import torch
import torch_geometric
from pathlib import Path
from scipy.spatial import Delaunay
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def compute_mesh_from_particles(
    particles: torch.Tensor,
    method: str = 'bpa',
    poisson_depth: int = 8,
    estimate_normals: bool = True,
    normal_radius: float = 0.01,
    normal_max_nn: int = 30,
    bpa_radii: list = None
) -> torch_geometric.data.Data:
    """Build a triangular mesh from particle positions.

    Args:
        particles: (N, 3) tensor of particle positions
        method: Meshing method - 'bpa' (Ball Pivoting, recommended), 'poisson', or 'delaunay'
        poisson_depth: Octree depth for Poisson reconstruction (default: 8)
        estimate_normals: Whether to estimate normals from point cloud
        normal_radius: Radius for normal estimation (relative to point cloud extent)
        normal_max_nn: Max nearest neighbors for normal estimation
        bpa_radii: List of radii for Ball Pivoting Algorithm. If None, auto-compute.

    Returns:
        mesh: torch_geometric.data.Data with pos, face, edge_index, norm attributes
    """
    import open3d as o3d

    device = particles.device
    particles_np = particles.cpu().numpy()

    if method == 'poisson':
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(particles_np)

        # Estimate normals if needed
        if estimate_normals or not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=normal_max_nn
                )
            )
            # Orient normals consistently (important for Poisson)
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson surface reconstruction
        print(f"  Running Poisson reconstruction (depth={poisson_depth})...")
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density vertices (artifacts far from surface)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)  # Remove bottom 10%
        vertices_to_remove = densities < density_threshold
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        # Clean up mesh
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()

        # Extract vertices and faces
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)

        print(f"  Poisson mesh: {len(vertices)} vertices, {len(faces)} faces")
        print(f"  ⚠️  Warning: Poisson adds {len(vertices) - len(particles_np)} extra vertices!")
        print(f"      This breaks correspondence with PGND particles.")

        # Convert to torch
        vertices_torch = torch.from_numpy(vertices).float().to(device)
        faces_torch = torch.from_numpy(faces).t().contiguous().to(device, dtype=torch.long)

    elif method == 'bpa':
        # Scipy 2D Delaunay (Open3D BPA segfaults in this environment)
        print(f"  Running 2D Delaunay triangulation (scipy)...")
        from scipy.spatial import Delaunay
        
        # Project to XZ plane for 2D Delaunay (cloth is roughly horizontal)
        # Project onto the two axes with highest variance (skip thin axis)
        variance = np.var(particles_np, axis=0)
        thin_axis = np.argmin(variance)
        proj_axes = [a for a in range(3) if a != thin_axis]
        print(f"  Thin axis: {['X','Y','Z'][thin_axis]} (var={variance[thin_axis]:.6f}), projecting onto {['X','Y','Z'][proj_axes[0]]}{['X','Y','Z'][proj_axes[1]]}")
        pts_2d = particles_np[:, proj_axes]
        tri = Delaunay(pts_2d)
        
        # Filter long-edge triangles (artifacts from convex hull)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=2).fit(particles_np)
        nn_dists, _ = knn.kneighbors(particles_np)
        avg_nn = nn_dists[:, 1].mean()
        threshold = avg_nn * 10
        
        edge_lens = np.linalg.norm(particles_np[tri.simplices[:, 0]] - particles_np[tri.simplices[:, 1]], axis=1)
        edge_lens = np.maximum(edge_lens, np.linalg.norm(particles_np[tri.simplices[:, 0]] - particles_np[tri.simplices[:, 2]], axis=1))
        edge_lens = np.maximum(edge_lens, np.linalg.norm(particles_np[tri.simplices[:, 1]] - particles_np[tri.simplices[:, 2]], axis=1))
        
        good_mask = edge_lens < threshold
        faces = tri.simplices[good_mask]
        
        unique_verts = len(np.unique(faces))
        print(f"  2D Delaunay: {len(faces)} triangles, {unique_verts}/{len(particles_np)} vertices ({100*unique_verts/len(particles_np):.1f}%)")
        
        vertices_torch = torch.from_numpy(particles_np).float().to(device)
        faces_torch = torch.from_numpy(faces).t().contiguous().to(device, dtype=torch.long)
        faces_torch = torch.from_numpy(faces).t().contiguous().to(device, dtype=torch.long)

    else:  # delaunay
        # Fallback: 2D Delaunay on XY projection
        print(f"  Using Delaunay triangulation...")
        pos_2d = particles_np[:, :2]
        tri = Delaunay(pos_2d, qhull_options='QJ')
        vertices_torch = particles
        faces_torch = torch.from_numpy(tri.simplices).t().contiguous().to(device, dtype=torch.long)

    # Create mesh data structure
    mesh = torch_geometric.data.Data(pos=vertices_torch, face=faces_torch)

    # Add edges (converts face connectivity to edge_index)
    mesh = torch_geometric.transforms.FaceToEdge(remove_faces=False)(mesh)

    # Compute face normals
    mesh = torch_geometric.transforms.GenerateMeshNormals()(mesh)

    return mesh


def compute_edge_features(mesh: torch_geometric.data.Data):
    """Compute edge displacement vectors and norms.

    Args:
        mesh: torch_geometric.data.Data with pos and edge_index

    Returns:
        edge_displacement: (num_edges, 3) relative displacement vectors
        edge_norm: (num_edges, 1) edge lengths
    """
    edge_index = mesh.edge_index
    pos = mesh.pos

    # Edge vectors: from source to target
    displacement = pos[edge_index[1]] - pos[edge_index[0]]
    norm = torch.norm(displacement, dim=1, keepdim=True)

    return displacement, norm


def save_mesh(mesh: torch_geometric.data.Data, output_path: Path):
    """Save mesh to .npz file.

    Args:
        mesh: torch_geometric.data.Data
        output_path: Path to save .npz file
    """
    np.savez_compressed(
        str(output_path),
        vertices=mesh.pos.cpu().numpy(),  # (N, 3)
        faces=mesh.face.cpu().numpy(),    # (3, num_faces)
        edge_index=mesh.edge_index.cpu().numpy(),  # (2, num_edges)
        normals=mesh.norm.cpu().numpy() if hasattr(mesh, 'norm') else None,
    )
    print(f"[build_cloth_mesh] Saved mesh to {output_path}")
    print(f"  - {mesh.pos.shape[0]} vertices")
    print(f"  - {mesh.face.shape[1]} faces")
    print(f"  - {mesh.edge_index.shape[1]} edges")


def save_mesh_ply(mesh: torch_geometric.data.Data, output_path: Path):
    """Save mesh as PLY file for visualization.

    Args:
        mesh: torch_geometric.data.Data
        output_path: Path to save .ply file
    """
    try:
        import open3d as o3d

        vertices = mesh.pos.cpu().numpy()
        faces = mesh.face.t().cpu().numpy()  # (num_faces, 3)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.compute_vertex_normals()

        o3d.io.write_triangle_mesh(str(output_path), mesh_o3d)
        print(f"[build_cloth_mesh] Saved visualization mesh to {output_path}")
    except ImportError:
        print("[build_cloth_mesh] open3d not available, skipping PLY export")


def load_pgnd_particles(episode_dir: Path, frame_idx: int = 0) -> torch.Tensor:
    """Load PGND particle positions for a specific frame.

    Args:
        episode_dir: Path to episode directory (contains traj.npz)
        frame_idx: Frame index to load (default: 0, first frame)

    Returns:
        particles: (N, 3) tensor of particle positions in preprocessed space
    """
    traj_path = episode_dir / 'traj.npz'
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Load trajectory (original world coordinates)
    traj_data = np.load(str(traj_path))
    xyz = traj_data['xyz']  # (T, N, 3)

    if frame_idx >= xyz.shape[0]:
        raise ValueError(f"Frame index {frame_idx} out of range (max: {xyz.shape[0]-1})")

    particles = torch.from_numpy(xyz[frame_idx]).float().cuda()
    return particles


def preprocess_particles(
    particles: torch.Tensor,
    preprocess_scale: float = 0.8,
    preprocess_with_table: bool = False,
    clip_bound: float = 0.05,
    num_grids: int = 50
) -> torch.Tensor:
    """Apply PGND's preprocessing transform to particles.

    This matches the preprocessing in PGND's dataset loader:
    1. Rotation (y-up → z-up)
    2. Scale
    3. Translation to center in [0,1]³

    Args:
        particles: (N, 3) tensor in original world coordinates
        preprocess_scale: Scale factor (default: 0.8)
        preprocess_with_table: If True, use table-aware centering
        clip_bound: Clipping boundary for table mode
        num_grids: Number of grid cells (for table mode)

    Returns:
        particles_preprocessed: (N, 3) tensor in [0,1]³ space
    """
    # Step 1: Rotation (y-up to z-up)
    R = torch.tensor(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        dtype=particles.dtype,
        device=particles.device
    )
    particles = particles @ R.T

    # Step 2: Scale
    particles = particles * preprocess_scale

    # Step 3: Translation
    dx = 1.0 / num_grids
    if preprocess_with_table:
        translation = torch.tensor([
            0.5 - (particles[:, 0].max() + particles[:, 0].min()) / 2,
            dx * (clip_bound + 0.5) + 1e-5 - particles[:, 1].min(),
            0.5 - (particles[:, 2].max() + particles[:, 2].min()) / 2,
        ], dtype=particles.dtype, device=particles.device)
    else:
        translation = torch.tensor([
            0.5 - (particles[:, 0].max() + particles[:, 0].min()) / 2,
            0.5 - (particles[:, 1].max() + particles[:, 1].min()) / 2,
            0.5 - (particles[:, 2].max() + particles[:, 2].min()) / 2,
        ], dtype=particles.dtype, device=particles.device)

    particles = particles + translation

    return particles


def build_mesh_for_episode(
    episode_dir: Path,
    output_dir: Path,
    frame_idx: int = 0,
    preprocess: bool = True,
    preprocess_scale: float = 0.8,
    save_ply: bool = True,
    method: str = 'bpa',
    poisson_depth: int = 8
):
    """Build and save mesh for a single episode.

    Args:
        episode_dir: Path to episode directory
        output_dir: Path to save mesh outputs
        frame_idx: Frame index to use for mesh topology (default: 0)
        preprocess: If True, apply PGND preprocessing to particles
        preprocess_scale: Scale factor for preprocessing
        save_ply: If True, save PLY file for visualization
        method: Meshing method - 'bpa' (recommended), 'poisson', or 'delaunay'
        poisson_depth: Octree depth for Poisson (6-10, default: 8)
    """
    print(f"[build_cloth_mesh] Building mesh for {episode_dir.name}")

    # Load particles
    particles = load_pgnd_particles(episode_dir, frame_idx)
    print(f"  Loaded {particles.shape[0]} particles from frame {frame_idx}")

    # Apply preprocessing if requested
    if preprocess:
        particles = preprocess_particles(particles, preprocess_scale=preprocess_scale)
        print(f"  Applied preprocessing (scale={preprocess_scale})")

    # Build mesh
    mesh = compute_mesh_from_particles(
        particles,
        method=method,
        poisson_depth=poisson_depth
    )

    # Compute edge features
    edge_displacement, edge_norm = compute_edge_features(mesh)
    mesh.edge_attr = torch.cat([edge_displacement, edge_norm], dim=1)

    # Save mesh
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mesh(mesh, output_dir / 'mesh.npz')

    if save_ply:
        save_mesh_ply(mesh, output_dir / 'mesh_viz.ply')

    return mesh


def main():
    parser = argparse.ArgumentParser(description='Build cloth mesh from PGND particles')
    parser.add_argument('--episode_dir', type=str, required=True,
                       help='Path to episode directory (contains traj.npz)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: episode_dir/mesh/)')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Frame index to use for mesh topology (default: 0)')
    parser.add_argument('--preprocess', action='store_true', default=True,
                       help='Apply PGND preprocessing transform')
    parser.add_argument('--preprocess_scale', type=float, default=0.8,
                       help='Preprocessing scale factor')
    parser.add_argument('--no_ply', action='store_true',
                       help='Skip PLY visualization file')
    parser.add_argument('--method', type=str, default='bpa',
                       choices=['bpa', 'poisson', 'delaunay'],
                       help='Meshing method: bpa (default, preserves vertices), poisson (adds vertices), delaunay (2D projection)')
    parser.add_argument('--poisson_depth', type=int, default=8,
                       help='Poisson octree depth (6-10, default: 8, only for --method=poisson)')

    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else episode_dir / 'mesh'

    build_mesh_for_episode(
        episode_dir=episode_dir,
        output_dir=output_dir,
        frame_idx=args.frame_idx,
        preprocess=args.preprocess,
        preprocess_scale=args.preprocess_scale,
        save_ply=not args.no_ply,
        method=args.method,
        poisson_depth=args.poisson_depth
    )

    print("[build_cloth_mesh] Done!")


if __name__ == '__main__':
    main()
