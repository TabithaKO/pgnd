"""
build_cloth_mesh_poisson.py — Enhanced Mesh Construction for Poisson Surfaces
==============================================================================

Extends build_cloth_mesh.py to properly handle Poisson surface reconstruction
for Ablation 2b. The key challenge: Poisson adds vertices that don't correspond
to PGND particles, but we need to deform those vertices when PGND predicts new
particle positions.

Solution: Compute interpolation weights from PGND particles to Poisson vertices
using K-nearest neighbors with RBF (Radial Basis Function) interpolation.

Usage:
    python build_cloth_mesh_poisson.py --episode_dir /path/to/episode --method poisson

Output:
    - mesh.npz: Poisson mesh (with extra vertices)
    - deformation_weights.npz: Interpolation weights to deform Poisson mesh from PGND particles
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from build_cloth_mesh import (
    load_pgnd_particles, preprocess_particles, compute_mesh_from_particles,
    compute_edge_features, save_mesh, save_mesh_ply
)


def compute_rbf_interpolation_weights(
    source_points: np.ndarray,
    target_points: np.ndarray,
    k_neighbors: int = 8,
    rbf_sigma: float = None
):
    """Compute RBF interpolation weights to deform target points from source points.

    Given source points (PGND particles) and target points (Poisson mesh vertices),
    compute weights W such that:
        target_pos_new ≈ W @ source_pos_new

    Uses K-nearest neighbor RBF interpolation for locality and efficiency.

    Args:
        source_points: (N_source, 3) PGND particle positions
        target_points: (N_target, 3) Poisson mesh vertex positions
        k_neighbors: Number of nearest source points to use per target point
        rbf_sigma: RBF kernel width. If None, auto-compute from distances.

    Returns:
        Dict with:
            - neighbor_indices: (N_target, k) indices of k nearest source points
            - weights: (N_target, k) RBF interpolation weights (sum to 1.0)
            - reconstruction_error: (N_target,) error of reconstructing target from source
    """
    from scipy.spatial import cKDTree

    print(f"[RBF interpolation] Computing weights for {len(target_points)} target points from {len(source_points)} source points...")

    # Build KDTree on source points
    tree = cKDTree(source_points)

    # Find k nearest source points for each target point
    distances, neighbor_indices = tree.query(target_points, k=k_neighbors)

    print(f"  K-NN distances: mean={distances.mean():.6f}, max={distances.max():.6f}")

    # Compute RBF weights
    if rbf_sigma is None:
        # Auto-compute sigma from median distance
        rbf_sigma = np.median(distances[:, 1:])  # Exclude self-distance
        print(f"  Auto-computed RBF sigma: {rbf_sigma:.6f}")

    # RBF kernel: w_ij = exp(-d_ij^2 / (2*sigma^2))
    weights = np.exp(-distances**2 / (2 * rbf_sigma**2))

    # Normalize weights to sum to 1.0
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / (weight_sums + 1e-8)

    # Validate reconstruction
    reconstructed_positions = np.zeros_like(target_points)
    for i in range(len(target_points)):
        neighbor_pos = source_points[neighbor_indices[i]]  # (k, 3)
        reconstructed_positions[i] = (weights[i:i+1].T * neighbor_pos).sum(axis=0)

    reconstruction_error = np.linalg.norm(reconstructed_positions - target_points, axis=1)

    print(f"  Reconstruction error: mean={reconstruction_error.mean():.6f}, max={reconstruction_error.max():.6f}")
    print(f"  Weight statistics: min={weights.min():.6f}, max={weights.max():.6f}")

    return {
        'neighbor_indices': neighbor_indices,
        'weights': weights,
        'reconstruction_error': reconstruction_error,
        'rbf_sigma': rbf_sigma
    }


def build_mesh_with_deformation_weights(
    episode_dir: Path,
    output_dir: Path,
    frame_idx: int = 0,
    preprocess: bool = True,
    preprocess_scale: float = 0.8,
    method: str = 'poisson',
    poisson_depth: int = 9,
    k_neighbors: int = 8,
    save_ply: bool = True
):
    """Build mesh and compute deformation weights for methods that add vertices.

    For BPA: No deformation weights needed (vertices = particles)
    For Poisson: Compute RBF interpolation weights (particles → vertices)

    Args:
        episode_dir: Path to episode directory
        output_dir: Path to save outputs
        frame_idx: Frame index for mesh topology
        preprocess: Whether to apply PGND preprocessing
        preprocess_scale: Preprocessing scale factor
        method: 'bpa' or 'poisson'
        poisson_depth: Octree depth for Poisson (recommend 9-10 for fine detail)
        k_neighbors: Number of neighbors for RBF interpolation
        save_ply: Whether to save PLY visualization
    """
    print(f"[build_cloth_mesh_poisson] Building mesh for {episode_dir.name}")
    print(f"  Method: {method}")

    # Load PGND particles (source points for deformation)
    particles = load_pgnd_particles(episode_dir, frame_idx)
    print(f"  Loaded {particles.shape[0]} PGND particles from frame {frame_idx}")

    # Store original particles (needed for interpolation weight computation)
    particles_original = particles.clone()

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

    mesh_vertices = mesh.pos.cpu().numpy()
    particles_np = particles.cpu().numpy()

    print(f"\n  Mesh: {len(mesh_vertices)} vertices, {mesh.face.shape[1]} faces")

    # Compute deformation weights if vertices != particles
    deformation_weights = None
    if len(mesh_vertices) != len(particles_np):
        print(f"\n  ⚠️  Mesh has {len(mesh_vertices) - len(particles_np)} extra vertices!")
        print(f"      Computing deformation weights (RBF interpolation)...")

        deformation_weights = compute_rbf_interpolation_weights(
            source_points=particles_np,
            target_points=mesh_vertices,
            k_neighbors=k_neighbors
        )

        print(f"  ✅ Deformation weights computed")
        print(f"     Mean reconstruction error: {deformation_weights['reconstruction_error'].mean():.6f}")

    else:
        print(f"  ✅ Vertex count matches particles ({len(mesh_vertices)})")
        print(f"     No deformation weights needed (1:1 correspondence)")

    # Compute edge features
    edge_displacement, edge_norm = compute_edge_features(mesh)
    mesh.edge_attr = torch.cat([edge_displacement, edge_norm], dim=1)

    # Save mesh
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mesh(mesh, output_dir / 'mesh.npz')

    # Save deformation weights
    if deformation_weights is not None:
        weights_path = output_dir / 'deformation_weights.npz'
        np.savez_compressed(
            str(weights_path),
            neighbor_indices=deformation_weights['neighbor_indices'],
            weights=deformation_weights['weights'],
            reconstruction_error=deformation_weights['reconstruction_error'],
            rbf_sigma=deformation_weights['rbf_sigma'],
            n_particles=len(particles_np),
            n_vertices=len(mesh_vertices),
            k_neighbors=k_neighbors
        )
        print(f"  Saved deformation weights to {weights_path}")

    # Save PLY
    if save_ply:
        save_mesh_ply(mesh, output_dir / 'mesh_viz.ply')

    print(f"\n✅ Mesh construction complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Vertices: {len(mesh_vertices)}")
    print(f"   Faces: {mesh.face.shape[1]}")
    if deformation_weights is not None:
        print(f"   Deformation: RBF interpolation from {len(particles_np)} particles")
    else:
        print(f"   Deformation: Direct (1:1 correspondence)")


def main():
    parser = argparse.ArgumentParser(
        description='Build cloth mesh with Poisson support and deformation weights'
    )
    parser.add_argument('--episode_dir', type=str, required=True,
                       help='Path to episode directory (contains traj.npz)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: episode_dir/mesh_<method>/)')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Frame index to use for mesh topology (default: 0)')
    parser.add_argument('--preprocess', action='store_true', default=True,
                       help='Apply PGND preprocessing transform')
    parser.add_argument('--preprocess_scale', type=float, default=0.8,
                       help='Preprocessing scale factor')
    parser.add_argument('--method', type=str, default='poisson',
                       choices=['bpa', 'poisson'],
                       help='Meshing method: bpa (preserves vertices) or poisson (adds vertices, default)')
    parser.add_argument('--poisson_depth', type=int, default=9,
                       help='Poisson octree depth (8-10, default: 9 for fine detail)')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='Number of neighbors for RBF interpolation (default: 8)')
    parser.add_argument('--no_ply', action='store_true',
                       help='Skip PLY visualization file')

    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else episode_dir / f'mesh_{args.method}'

    build_mesh_with_deformation_weights(
        episode_dir=episode_dir,
        output_dir=output_dir,
        frame_idx=args.frame_idx,
        preprocess=args.preprocess,
        preprocess_scale=args.preprocess_scale,
        method=args.method,
        poisson_depth=args.poisson_depth,
        k_neighbors=args.k_neighbors,
        save_ply=not args.no_ply
    )

    print("\n[build_cloth_mesh_poisson] Done!")


if __name__ == '__main__':
    main()
