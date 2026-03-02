"""
test_mesh_gaussian_model_synthetic.py — Synthetic test for MeshGaussianModel
==============================================================================

Tests the mesh-constrained Gaussian model with synthetic data to verify
the architecture works without requiring specific data files.

Usage:
    python test_mesh_gaussian_model_synthetic.py
"""

import numpy as np
import torch
import torch_geometric.data
from pathlib import Path

from mesh_gaussian_model import MeshGaussianModel


def create_synthetic_mesh(n_vertices=100, device='cuda'):
    """Create a synthetic mesh for testing."""
    print("Creating synthetic mesh...")

    # Random point cloud
    points = torch.randn(n_vertices, 3, device=device) * 0.3 + 0.5

    # Simple Delaunay triangulation (2D projection for simplicity)
    from scipy.spatial import Delaunay
    points_2d = points[:, [0, 2]].cpu().numpy()
    tri = Delaunay(points_2d)

    # Create mesh data
    mesh = torch_geometric.data.Data()
    mesh.pos = points
    mesh.face = torch.from_numpy(tri.simplices.T).long().to(device)

    # Compute edges
    from torch_geometric.utils import to_undirected
    edge_index = torch.cat([
        torch.stack([mesh.face[0], mesh.face[1]]),
        torch.stack([mesh.face[1], mesh.face[2]]),
        torch.stack([mesh.face[2], mesh.face[0]]),
    ], dim=1)
    mesh.edge_index = to_undirected(edge_index)

    print(f"  Created mesh: {n_vertices} vertices, {mesh.face.shape[1]} faces")
    return mesh


def create_synthetic_gaussians(n_gaussians=500, device='cuda'):
    """Create synthetic Gaussian parameters."""
    print("Creating synthetic Gaussians...")

    positions = torch.randn(n_gaussians, 3, device=device) * 0.3 + 0.5
    colors = torch.rand(n_gaussians, 3, device=device)
    scales = torch.rand(n_gaussians, 3, device=device) * 0.01 + 0.005
    quats = torch.randn(n_gaussians, 4, device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    opacities = torch.rand(n_gaussians, 1, device=device) * 0.5 + 0.5

    print(f"  Created {n_gaussians} Gaussians")

    return {
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'quats': quats,
        'opacities': opacities
    }


def assign_gaussians_to_faces(mesh, gaussian_positions):
    """Assign each Gaussian to nearest face with barycentric coords."""
    print("Assigning Gaussians to faces...")

    from scipy.spatial import cKDTree

    # Compute face centers
    face_vertices = mesh.pos[mesh.face.T].cpu().numpy()  # (N_faces, 3, 3)
    face_centers = face_vertices.mean(axis=1)  # (N_faces, 3)

    # Find nearest face for each Gaussian
    tree = cKDTree(face_centers)
    gaussian_pos_np = gaussian_positions.cpu().numpy()
    distances, face_ids = tree.query(gaussian_pos_np, k=1)

    # Compute barycentric coordinates (use simple 1/3, 1/3, 1/3 for now)
    n_gauss = len(gaussian_positions)
    bary_coords = np.ones((n_gauss, 3)) / 3.0

    print(f"  Assigned {n_gauss} Gaussians to {len(face_centers)} faces")
    print(f"  Mean distance to face: {distances.mean():.6f}")

    return face_ids, bary_coords


def test_manual_initialization():
    """Test 1: Manual initialization with synthetic data."""
    print("="*80)
    print("TEST 1: Manual Initialization with Synthetic Data")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create synthetic data
    mesh = create_synthetic_mesh(n_vertices=100, device=device)
    gs_data = create_synthetic_gaussians(n_gaussians=500, device=device)
    face_ids, bary_coords = assign_gaussians_to_faces(mesh, gs_data['positions'])

    # Initialize model manually
    model = MeshGaussianModel(sh_degree=0)
    model.mesh = mesh
    model.face_ids = torch.from_numpy(face_ids).long().to(device)
    model.face_bary = torch.nn.Parameter(
        torch.from_numpy(bary_coords).float().to(device).requires_grad_(True)
    )
    model._colors = torch.nn.Parameter(gs_data['colors'].requires_grad_(True))
    model._scales = torch.nn.Parameter(torch.log(gs_data['scales']).requires_grad_(True))
    model._rotations = torch.nn.Parameter(gs_data['quats'].requires_grad_(True))
    model._opacities = torch.nn.Parameter(
        torch.logit(gs_data['opacities'], eps=1e-8).requires_grad_(True)
    )
    model.use_rbf_deformation = False

    print(f"\n✅ Manual initialization successful!")
    print(f"   Gaussians: {model.num_gaussians}")
    print(f"   Mesh vertices: {model.mesh.pos.shape[0]}")
    print(f"   Mesh faces: {model.mesh.face.shape[1]}")

    return model


def test_forward_pass(model):
    """Test 2: Forward pass."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass")
    print("="*80)

    # Original mesh
    xyz = model.get_xyz(deformed_vertices=None)
    rotations = model.get_rotation(deformed_vertices=None)
    colors = model.get_colors()
    scales = model.get_scales()
    opacities = model.get_opacities()

    print(f"\n✅ Forward pass successful!")
    print(f"   Positions: {xyz.shape}, range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"   Rotations: {rotations.shape}")
    print(f"   Colors: {colors.shape}, range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"   Scales: {scales.shape}, range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"   Opacities: {opacities.shape}, range: [{opacities.min():.3f}, {opacities.max():.3f}]")

    # Verify shapes
    assert xyz.shape == (model.num_gaussians, 3)
    assert rotations.shape == (model.num_gaussians, 4)
    assert colors.shape == (model.num_gaussians, 3)
    assert scales.shape == (model.num_gaussians, 3)
    assert opacities.shape == (model.num_gaussians, 1)

    # Verify ranges
    assert 0.0 <= colors.min() and colors.max() <= 1.0, "Colors out of range"
    assert 0.0 <= opacities.min() and opacities.max() <= 1.0, "Opacities out of range"
    assert scales.min() > 0.0, "Scales must be positive"

    return xyz


def test_deformation(model, original_xyz):
    """Test 3: Mesh deformation."""
    print("\n" + "="*80)
    print("TEST 3: Mesh Deformation")
    print("="*80)

    # Perturb mesh vertices
    perturbation = torch.randn_like(model.mesh.pos) * 0.02
    deformed_vertices = model.mesh.pos + perturbation

    # Get deformed Gaussians
    deformed_xyz = model.get_xyz(deformed_vertices=deformed_vertices)
    deformed_rotations = model.get_rotation(deformed_vertices=deformed_vertices)

    # Measure displacement
    displacement = (deformed_xyz - original_xyz).norm(dim=1)

    print(f"\n✅ Deformation successful!")
    print(f"   Mean displacement: {displacement.mean():.6f}")
    print(f"   Max displacement: {displacement.max():.6f}")
    print(f"   Deformed rotations: {deformed_rotations.shape}")

    assert displacement.mean() > 0.0, "Gaussians should move when mesh deforms"


def test_gradient_flow(model):
    """Test 4: Gradient flow through deformation."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow")
    print("="*80)

    # Test gradient through deformation
    deformed_vertices = model.mesh.pos.clone().detach().requires_grad_(True)
    xyz = model.get_xyz(deformed_vertices=deformed_vertices)
    loss = xyz.sum()
    loss.backward()

    assert deformed_vertices.grad is not None, "Gradients should flow to vertices"
    print(f"\n✅ Gradient flow (vertices): {deformed_vertices.grad.norm():.6f}")

    # Test gradient through GS parameters
    model.zero_grad()
    xyz = model.get_xyz()
    colors = model.get_colors()
    loss = xyz.sum() + colors.sum()
    loss.backward()

    assert model.face_bary.grad is not None
    assert model._colors.grad is not None

    print(f"   Gradient flow (face_bary): {model.face_bary.grad.norm():.6f}")
    print(f"   Gradient flow (colors): {model._colors.grad.norm():.6f}")


def test_optimizer(model):
    """Test 5: Optimizer setup and step."""
    print("\n" + "="*80)
    print("TEST 5: Optimizer Setup")
    print("="*80)

    param_groups = model.get_optimizer_param_groups()
    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    print(f"\n✅ Optimizer created!")
    print(f"   Parameter groups: {len(param_groups)}")
    for group in param_groups:
        print(f"     {group['name']}: lr={group['lr']:.2e}")

    # Run gradient step
    model.zero_grad()
    xyz = model.get_xyz()
    loss = xyz.mean()
    loss.backward()
    optimizer.step()

    print(f"   ✅ Gradient step completed")


def test_save_load(model):
    """Test 6: Save and load."""
    print("\n" + "="*80)
    print("TEST 6: Save/Load")
    print("="*80)

    save_dir = Path('experiments/log/mesh_gaussian_test_synthetic')

    # Get original state
    original_xyz = model.get_xyz().detach().cpu().numpy()
    original_colors = model.get_colors().detach().cpu().numpy()

    # Save
    model.save(save_dir)
    print(f"   Saved to {save_dir}")

    # Load into new model
    model2 = MeshGaussianModel(sh_degree=0)
    model2.load(save_dir, device=model.mesh.pos.device)

    # Compare
    loaded_xyz = model2.get_xyz().detach().cpu().numpy()
    loaded_colors = model2.get_colors().detach().cpu().numpy()

    xyz_diff = np.abs(original_xyz - loaded_xyz).max()
    color_diff = np.abs(original_colors - loaded_colors).max()

    print(f"\n✅ Save/load successful!")
    print(f"   Max position diff: {xyz_diff:.6e}")
    print(f"   Max color diff: {color_diff:.6e}")

    assert xyz_diff < 1e-5
    assert color_diff < 1e-5


def main():
    print("Testing MeshGaussianModel with synthetic data\n")

    try:
        model = test_manual_initialization()
        original_xyz = test_forward_pass(model)
        test_deformation(model, original_xyz)
        test_gradient_flow(model)
        test_optimizer(model)
        test_save_load(model)

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nMeshGaussianModel architecture is validated!")
        print("Ready for integration with real PGND data.")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
