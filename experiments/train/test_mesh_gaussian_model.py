"""
test_mesh_gaussian_model.py — Unit test for MeshGaussianModel
===============================================================

Tests the mesh-constrained Gaussian model initialization and deformation.

Usage:
    python test_mesh_gaussian_model.py --episode_dir /path/to/episode --splat_path /path/to/file.splat
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from mesh_gaussian_model import MeshGaussianModel


def test_initialization(episode_dir: Path, splat_path: Path, method: str = 'bpa'):
    """Test model initialization from episode data."""
    print("="*80)
    print("TEST 1: Model Initialization")
    print("="*80)

    model = MeshGaussianModel(sh_degree=0)
    model.initialize_from_episode(
        episode_dir=episode_dir,
        splat_path=splat_path,
        method=method,
        frame_idx=0,
        preprocess=True,
        preprocess_scale=0.8,
    )

    print(f"\n✅ Initialization successful!")
    print(f"   Gaussians: {model.num_gaussians}")
    print(f"   Mesh vertices: {model.mesh.pos.shape[0]}")
    print(f"   Mesh faces: {model.mesh.face.shape[1]}")
    print(f"   Using RBF: {model.use_rbf_deformation}")

    return model


def test_forward_pass(model: MeshGaussianModel):
    """Test forward pass: get Gaussian positions, rotations, etc."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass (Original Mesh)")
    print("="*80)

    # Get Gaussian attributes with original mesh (no deformation)
    xyz = model.get_xyz(deformed_vertices=None)
    rotations = model.get_rotation(deformed_vertices=None)
    colors = model.get_colors()
    scales = model.get_scales()
    opacities = model.get_opacities()

    print(f"\n✅ Forward pass successful!")
    print(f"   Gaussian positions: {xyz.shape}")
    print(f"   Gaussian rotations: {rotations.shape}")
    print(f"   Gaussian colors: {colors.shape} (range: [{colors.min():.3f}, {colors.max():.3f}])")
    print(f"   Gaussian scales: {scales.shape} (range: [{scales.min():.6f}, {scales.max():.6f}])")
    print(f"   Gaussian opacities: {opacities.shape} (range: [{opacities.min():.3f}, {opacities.max():.3f}])")

    # Check output shapes
    assert xyz.shape == (model.num_gaussians, 3), "XYZ shape mismatch"
    assert rotations.shape == (model.num_gaussians, 4), "Rotations shape mismatch"
    assert colors.shape == (model.num_gaussians, 3), "Colors shape mismatch"
    assert scales.shape == (model.num_gaussians, 3), "Scales shape mismatch"
    assert opacities.shape == (model.num_gaussians, 1), "Opacities shape mismatch"

    # Check value ranges
    assert colors.min() >= 0.0 and colors.max() <= 1.0, "Colors out of [0, 1] range"
    assert opacities.min() >= 0.0 and opacities.max() <= 1.0, "Opacities out of [0, 1] range"
    assert scales.min() > 0.0, "Scales must be positive"

    return xyz


def test_deformation(model: MeshGaussianModel, original_xyz: torch.Tensor):
    """Test mesh deformation: perturb vertices and check Gaussian positions update."""
    print("\n" + "="*80)
    print("TEST 3: Mesh Deformation")
    print("="*80)

    # Create a small random perturbation to mesh vertices
    n_vertices = model.mesh.pos.shape[0]
    perturbation = torch.randn(n_vertices, 3, device=model.mesh.pos.device) * 0.01

    deformed_vertices = model.mesh.pos + perturbation

    # Get deformed Gaussian positions
    deformed_xyz = model.get_xyz(deformed_vertices=deformed_vertices)
    deformed_rotations = model.get_rotation(deformed_vertices=deformed_vertices)

    # Check that positions changed
    position_diff = (deformed_xyz - original_xyz).norm(dim=1)
    mean_displacement = position_diff.mean().item()
    max_displacement = position_diff.max().item()

    print(f"\n✅ Deformation successful!")
    print(f"   Mean Gaussian displacement: {mean_displacement:.6f}")
    print(f"   Max Gaussian displacement: {max_displacement:.6f}")
    print(f"   Deformed rotations: {deformed_rotations.shape}")

    assert mean_displacement > 0.0, "Gaussians should move when mesh deforms"
    print(f"   ✅ Gaussians moved as expected")


def test_gradient_flow(model: MeshGaussianModel):
    """Test that gradients flow through deformation."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow")
    print("="*80)

    # Create deformed vertices with requires_grad=True
    deformed_vertices = model.mesh.pos.clone().detach().requires_grad_(True)

    # Forward pass
    xyz = model.get_xyz(deformed_vertices=deformed_vertices)
    rotations = model.get_rotation(deformed_vertices=deformed_vertices)

    # Dummy loss: sum of positions
    loss = xyz.sum()
    loss.backward()

    # Check gradients exist
    assert deformed_vertices.grad is not None, "Gradients should flow to deformed_vertices"
    grad_norm = deformed_vertices.grad.norm().item()

    print(f"\n✅ Gradient flow successful!")
    print(f"   Gradient norm (deformed_vertices): {grad_norm:.6f}")

    # Check that GS parameters have gradients
    model.zero_grad()
    # Need to recompute xyz since the graph was freed after first backward
    xyz_new = model.get_xyz()
    colors = model.get_colors()
    loss = xyz_new.sum() + colors.sum()
    loss.backward()

    print(f"   Gradient norm (face_bary): {model.face_bary.grad.norm().item():.6f}")
    print(f"   Gradient norm (colors): {model._colors.grad.norm().item():.6f}")

    assert model.face_bary.grad is not None, "face_bary should have gradients"
    assert model._colors.grad is not None, "colors should have gradients"


def test_optimizer_setup(model: MeshGaussianModel):
    """Test optimizer parameter groups."""
    print("\n" + "="*80)
    print("TEST 5: Optimizer Setup")
    print("="*80)

    param_groups = model.get_optimizer_param_groups(
        lr_position=1e-4,
        lr_color=2.5e-3,
        lr_scale=5e-3,
        lr_rotation=1e-3,
        lr_opacity=5e-2,
    )

    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    print(f"\n✅ Optimizer setup successful!")
    print(f"   Parameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"     {i+1}. {group['name']}: lr={group['lr']:.2e}, params={len(group['params'])}")

    # Sanity check: run a gradient step
    model.zero_grad()
    xyz = model.get_xyz()
    loss = xyz.mean()
    loss.backward()
    optimizer.step()

    print(f"   ✅ Gradient step completed")


def test_save_load(model: MeshGaussianModel, save_dir: Path):
    """Test save and load."""
    print("\n" + "="*80)
    print("TEST 6: Save/Load")
    print("="*80)

    # Get original state
    original_xyz = model.get_xyz().detach().cpu().numpy()
    original_colors = model.get_colors().detach().cpu().numpy()

    # Save
    model.save(save_dir)
    print(f"   Saved to {save_dir}")

    # Create new model and load
    model2 = MeshGaussianModel(sh_degree=0)
    model2.load(save_dir, device='cuda')

    # Compare
    loaded_xyz = model2.get_xyz().detach().cpu().numpy()
    loaded_colors = model2.get_colors().detach().cpu().numpy()

    xyz_diff = np.abs(original_xyz - loaded_xyz).max()
    color_diff = np.abs(original_colors - loaded_colors).max()

    print(f"\n✅ Save/load successful!")
    print(f"   Max position difference: {xyz_diff:.6e}")
    print(f"   Max color difference: {color_diff:.6e}")

    assert xyz_diff < 1e-5, "XYZ should match after save/load"
    assert color_diff < 1e-5, "Colors should match after save/load"


def main():
    parser = argparse.ArgumentParser(description='Test MeshGaussianModel')
    parser.add_argument('--episode_dir', type=str, required=False,
                       default='experiments/log/data/1109_cloth_gello_cali2_processed/sub_episodes_v/episode_0162',
                       help='Path to episode directory')
    parser.add_argument('--splat_path', type=str, required=False,
                       help='Path to .splat file')
    parser.add_argument('--method', type=str, default='bpa',
                       choices=['bpa', 'poisson'],
                       help='Meshing method')
    parser.add_argument('--save_dir', type=str, default='experiments/log/mesh_gaussian_test',
                       help='Directory to save test outputs')

    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)

    # Auto-find splat file if not provided
    if args.splat_path:
        splat_path = Path(args.splat_path)
    else:
        # Look for .splat in episode's gs/ directory
        # episode_dir example: log/data/recording_processed/sub_episodes_v/episode_0162
        # splat path: log/data_cloth/recording_processed/episode_0162/gs/*.splat
        recording_name = episode_dir.parent.parent.name  # e.g. '1109_cloth_gello_cali2_processed'
        episode_name = episode_dir.name  # e.g. 'episode_0162'

        # Try to find in data_cloth
        data_cloth_dir = Path('experiments/log/data_cloth') / recording_name / episode_name / 'gs'
        if data_cloth_dir.exists():
            splat_files = sorted(data_cloth_dir.glob('*.splat'))
            if splat_files:
                splat_path = splat_files[0]
                print(f"Auto-detected splat file: {splat_path}")
            else:
                print(f"ERROR: No .splat files found in {data_cloth_dir}")
                return
        else:
            print(f"ERROR: data_cloth directory not found: {data_cloth_dir}")
            print("Please specify --splat_path manually")
            return

    if not episode_dir.exists():
        print(f"ERROR: Episode directory not found: {episode_dir}")
        return
    if not splat_path.exists():
        print(f"ERROR: Splat file not found: {splat_path}")
        return

    save_dir = Path(args.save_dir)

    try:
        # Run tests
        model = test_initialization(episode_dir, splat_path, method=args.method)
        original_xyz = test_forward_pass(model)
        test_deformation(model, original_xyz)
        test_gradient_flow(model)
        test_optimizer_setup(model)
        test_save_load(model, save_dir)

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print(f"\nMeshGaussianModel is ready for training!")
        print(f"  Method: {args.method}")
        print(f"  Gaussians: {model.num_gaussians}")
        print(f"  Mesh vertices: {model.mesh.pos.shape[0]}")
        print(f"  Using RBF deformation: {model.use_rbf_deformation}")

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
