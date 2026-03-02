"""
anchor_gaussians_to_mesh.py — Anchor Gaussians to Mesh Faces
==============================================================

For each Gaussian in a .splat file, find the nearest mesh face and compute
barycentric coordinates. This creates the anchoring needed for mesh-constrained
Gaussian Splatting in Ablation 2a.

Usage:
    python anchor_gaussians_to_mesh.py --episode_dir /path/to/episode_XXXX --splat_path /path/to/file.splat

Output:
    - gaussian_anchors.npz: Contains face_ids, bary_coords for each Gaussian
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from build_cloth_mesh import build_mesh_for_episode

# Import read_splat_raw from render_loss.py
try:
    from render_loss import read_splat_raw
except ImportError:
    # Define it here if not available
    def read_splat_raw(path: str):
        """Read a .splat file into numpy arrays."""
        import struct
        data = Path(path).read_bytes()
        n = len(data) // 32

        dt = np.dtype([
            ('pos', np.float32, 3),
            ('scale', np.float32, 3),
            ('rgba', np.uint8, 4),
            ('rot', np.uint8, 4),
        ])
        arr = np.frombuffer(data, dtype=dt, count=n)

        pts = arr['pos'].copy()
        scales = arr['scale'].copy()
        colors = arr['rgba'][:, :3].astype(np.float32) / 255.0
        opacities = arr['rgba'][:, 3:4].astype(np.float32) / 255.0

        quats_raw = arr['rot'].astype(np.float32)
        quats = (quats_raw / 128.0) - 1.0
        qnorm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / (qnorm + 1e-8)

        return {
            'pts': pts,
            'colors': colors,
            'scales': scales,
            'quats': quats,
            'opacities': opacities,
        }


def compute_barycentric_coordinates(point, v0, v1, v2):
    """Compute barycentric coordinates of a point relative to a triangle.

    Given a point P and triangle vertices v0, v1, v2, compute weights (b0, b1, b2)
    such that: P ≈ b0*v0 + b1*v1 + b2*v2, with b0 + b1 + b2 = 1

    Args:
        point: (3,) point position
        v0, v1, v2: (3,) triangle vertices

    Returns:
        bary: (3,) barycentric coordinates [b0, b1, b2]
    """
    # Vectors from v0 to v1 and v0 to v2
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Vector from v0 to point
    vec = point - v0

    # Solve the linear system using Cramer's rule
    # [edge1, edge2] @ [b1, b2]^T = vec
    # where b0 = 1 - b1 - b2

    d00 = np.dot(edge1, edge1)
    d01 = np.dot(edge1, edge2)
    d11 = np.dot(edge2, edge2)
    d20 = np.dot(vec, edge1)
    d21 = np.dot(vec, edge2)

    denom = d00 * d11 - d01 * d01

    if abs(denom) < 1e-10:
        # Degenerate triangle, use equal weights
        return np.array([1/3, 1/3, 1/3])

    b1 = (d11 * d20 - d01 * d21) / denom
    b2 = (d00 * d21 - d01 * d20) / denom
    b0 = 1.0 - b1 - b2

    return np.array([b0, b1, b2])


def find_nearest_face_kdtree(gaussian_positions, mesh_vertices, mesh_faces):
    """Find nearest mesh face for each Gaussian using KDTree for face centers.

    Args:
        gaussian_positions: (N_gauss, 3) Gaussian positions
        mesh_vertices: (N_verts, 3) mesh vertex positions
        mesh_faces: (3, N_faces) mesh face indices

    Returns:
        nearest_face_ids: (N_gauss,) index of nearest face for each Gaussian
    """
    from scipy.spatial import cKDTree

    # Compute face centers
    face_vertices = mesh_vertices[mesh_faces.T]  # (N_faces, 3, 3)
    face_centers = face_vertices.mean(axis=1)  # (N_faces, 3)

    # Build KDTree on face centers
    tree = cKDTree(face_centers)

    # Query nearest face for each Gaussian
    distances, nearest_face_ids = tree.query(gaussian_positions, k=1)

    return nearest_face_ids, distances


def anchor_gaussians_to_mesh(
    splat_path: Path,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    preprocess_params: dict = None
):
    """Anchor each Gaussian to its nearest mesh face with barycentric coordinates.

    Args:
        splat_path: Path to .splat file
        mesh_vertices: (N_verts, 3) mesh vertex positions in preprocessed [0,1]³ space
        mesh_faces: (3, N_faces) mesh face indices
        preprocess_params: Dict with 'scale', 'rotation', 'translation' for coordinate transforms

    Returns:
        anchors: Dict with:
            - face_ids: (N_gauss,) nearest face index for each Gaussian
            - bary_coords: (N_gauss, 3) barycentric coordinates
            - gaussian_positions: (N_gauss, 3) original Gaussian positions
            - distances: (N_gauss,) distance to nearest face center
    """
    print(f"[anchor_gaussians] Loading Gaussians from {splat_path.name}")

    # Load Gaussian Splatting data
    gs_data = read_splat_raw(str(splat_path))
    gaussian_positions = gs_data['pts']  # (N_gauss, 3) in original world coords
    n_gaussians = len(gaussian_positions)

    print(f"  Loaded {n_gaussians} Gaussians")

    # Transform Gaussian positions to preprocessed space if needed
    if preprocess_params is not None:
        print(f"  Transforming Gaussians to preprocessed space...")
        # Apply same transform as PGND particles
        R = preprocess_params['rotation']
        scale = preprocess_params['scale']
        translation = preprocess_params['translation']

        gaussian_positions = gaussian_positions @ R.T
        gaussian_positions = gaussian_positions * scale
        gaussian_positions = gaussian_positions + translation

    print(f"  Gaussian position range:")
    print(f"    X: [{gaussian_positions[:, 0].min():.4f}, {gaussian_positions[:, 0].max():.4f}]")
    print(f"    Y: [{gaussian_positions[:, 1].min():.4f}, {gaussian_positions[:, 1].max():.4f}]")
    print(f"    Z: [{gaussian_positions[:, 2].min():.4f}, {gaussian_positions[:, 2].max():.4f}]")

    # Find nearest face for each Gaussian
    print(f"  Finding nearest face for each Gaussian...")
    nearest_face_ids, distances = find_nearest_face_kdtree(
        gaussian_positions, mesh_vertices, mesh_faces
    )

    print(f"  Distance to nearest face (mean): {distances.mean():.6f}")
    print(f"  Distance to nearest face (max): {distances.max():.6f}")

    # Compute barycentric coordinates for each Gaussian
    print(f"  Computing barycentric coordinates...")
    bary_coords = np.zeros((n_gaussians, 3))

    for i, (gauss_pos, face_id) in enumerate(zip(gaussian_positions, nearest_face_ids)):
        # Get the 3 vertices of this face
        v0_idx, v1_idx, v2_idx = mesh_faces[:, face_id]
        v0 = mesh_vertices[v0_idx]
        v1 = mesh_vertices[v1_idx]
        v2 = mesh_vertices[v2_idx]

        # Compute barycentric coordinates
        bary = compute_barycentric_coordinates(gauss_pos, v0, v1, v2)
        bary_coords[i] = bary

        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_gaussians}", end='\r')

    print(f"    Progress: {n_gaussians}/{n_gaussians}")

    # Validate barycentric coordinates
    bary_sums = bary_coords.sum(axis=1)
    print(f"\n  Barycentric coordinate validation:")
    print(f"    Sum should be ~1.0: mean={bary_sums.mean():.6f}, std={bary_sums.std():.6f}")
    print(f"    Out of range (<0 or >1): {np.sum((bary_coords < -0.1) | (bary_coords > 1.1))}")

    # Check reconstruction error
    # Reconstruct Gaussian positions from barycentric coords and compare to original
    reconstructed_positions = np.zeros_like(gaussian_positions)
    for i, (bary, face_id) in enumerate(zip(bary_coords, nearest_face_ids)):
        v0_idx, v1_idx, v2_idx = mesh_faces[:, face_id]
        v0 = mesh_vertices[v0_idx]
        v1 = mesh_vertices[v1_idx]
        v2 = mesh_vertices[v2_idx]
        reconstructed_positions[i] = bary[0]*v0 + bary[1]*v1 + bary[2]*v2

    reconstruction_error = np.linalg.norm(reconstructed_positions - gaussian_positions, axis=1)
    print(f"  Reconstruction error (should match distance to face):")
    print(f"    Mean: {reconstruction_error.mean():.6f}")
    print(f"    Max: {reconstruction_error.max():.6f}")

    return {
        'face_ids': nearest_face_ids,
        'bary_coords': bary_coords,
        'gaussian_positions': gaussian_positions,
        'distances': distances,
        'reconstruction_error': reconstruction_error
    }


def main():
    parser = argparse.ArgumentParser(description='Anchor Gaussians to mesh faces')
    parser.add_argument('--episode_dir', type=str, required=True,
                       help='Path to episode directory')
    parser.add_argument('--splat_path', type=str, required=True,
                       help='Path to .splat file')
    parser.add_argument('--mesh_dir', type=str, default=None,
                       help='Path to mesh directory (default: episode_dir/mesh_bpa/)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: mesh_dir/)')
    parser.add_argument('--method', type=str, default='bpa',
                       choices=['bpa', 'poisson', 'delaunay'],
                       help='Meshing method (default: bpa)')

    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    splat_path = Path(args.splat_path)

    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    if not splat_path.exists():
        raise FileNotFoundError(f"Splat file not found: {splat_path}")

    # Determine mesh and output directories
    if args.mesh_dir:
        mesh_dir = Path(args.mesh_dir)
    else:
        mesh_dir = episode_dir / f'mesh_{args.method}'

    output_dir = Path(args.output_dir) if args.output_dir else mesh_dir

    # Build or load mesh
    mesh_npz_path = mesh_dir / 'mesh.npz'
    if not mesh_npz_path.exists():
        print(f"[anchor_gaussians] Mesh not found, building...")
        build_mesh_for_episode(
            episode_dir=episode_dir,
            output_dir=mesh_dir,
            frame_idx=0,
            method=args.method
        )
    else:
        print(f"[anchor_gaussians] Loading existing mesh from {mesh_npz_path}")

    # Load mesh
    mesh_data = np.load(str(mesh_npz_path))
    mesh_vertices = mesh_data['vertices']
    mesh_faces = mesh_data['faces']

    print(f"  Mesh: {len(mesh_vertices)} vertices, {mesh_faces.shape[1]} faces")

    # Get preprocessing parameters (needed to transform Gaussians to mesh space)
    # For now, assume Gaussians are already in the same space as mesh
    # TODO: Load actual preprocessing params from episode metadata
    preprocess_params = {
        'rotation': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # y-up to z-up
        'scale': 0.8,
        'translation': np.array([0.0, 0.0, 0.0])  # Computed from centering
    }

    # Anchor Gaussians to mesh
    anchors = anchor_gaussians_to_mesh(
        splat_path=splat_path,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        preprocess_params=preprocess_params
    )

    # Save anchors
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'gaussian_anchors.npz'

    np.savez_compressed(
        str(output_path),
        face_ids=anchors['face_ids'],
        bary_coords=anchors['bary_coords'],
        gaussian_positions=anchors['gaussian_positions'],
        distances=anchors['distances'],
        reconstruction_error=anchors['reconstruction_error']
    )

    print(f"\n✅ Saved Gaussian anchors to {output_path}")
    print(f"   {len(anchors['face_ids'])} Gaussians anchored to mesh")
    print(f"   Mean distance to face: {anchors['distances'].mean():.6f}")
    print(f"   Mean reconstruction error: {anchors['reconstruction_error'].mean():.6f}")
    print("\n[anchor_gaussians] Done!")


if __name__ == '__main__':
    main()
