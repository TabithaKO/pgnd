"""
render_loss.py — Observation-Space Loss for PGND Dynamics Training
==================================================================

Adds a differentiable Gaussian Splatting rendering loss to the PGND training
loop, backpropagating image-space gradients through:

    image loss → GS rasterizer → LBS deformation → particle positions → fθ

The GS model parameters (colors, scales, opacities, rotations) are FROZEN.
Only the dynamics model fθ receives gradients from this loss.

Usage:
    Place this file in ~/pgnd/experiments/train/render_loss.py

    In train_eval.py, the training loop accumulates:
        loss = loss_geometry + λ_render * loss_render

Architecture:
    1. Load a pre-trained GS reconstruction (.splat) for the first frame
    2. At each rollout step, use LBS to deform GS kernels based on
       predicted particle displacements
    3. Rasterize the deformed Gaussians from known camera viewpoints
    4. Compare rendered image to ground-truth RGB observation
    5. Backprop through the differentiable rasterizer → LBS → particles → fθ

Key design decisions:
    - GS params are frozen (no grad) — we only train the dynamics model
    - LBS weights are precomputed per-episode (amortized cost)
    - We render from a single camera per step (configurable)
    - Image loss = L1 + λ_ssim * (1 - SSIM), matching PhysTwin's formulation
    - Render loss is computed every `render_every_n_steps` rollout steps
      to manage GPU memory and compute cost

Dependencies (already in pgnd env):
    - diff_gaussian_rasterization (installed from third-party/)
    - kornia (for SSIM, quaternion ops)
    - sklearn (for KNN in LBS weight computation)
"""

import struct
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


# =============================================================================
# Splat I/O (matches PGND's format: 32 bytes/gaussian)
# =============================================================================

def read_splat_raw(path: str) -> Dict[str, np.ndarray]:
    """Read a .splat file into numpy arrays.
    
    Format: 32 bytes per gaussian
        - 3 floats: position (x, y, z)
        - 3 floats: scale (sx, sy, sz)  
        - 4 uint8: RGBA color
        - 4 uint8: quaternion rotation (encoded as 0-255 -> mapped to floats)
    """
    data = Path(path).read_bytes()
    n = len(data) // 32
    
    # Parse all at once using numpy
    dt = np.dtype([
        ('pos', np.float32, 3),
        ('scale', np.float32, 3),
        ('rgba', np.uint8, 4),
        ('rot', np.uint8, 4),
    ])
    arr = np.frombuffer(data, dtype=dt, count=n)
    
    pts = arr['pos'].copy()
    scales = arr['scale'].copy()
    
    # Colors: uint8 [0,255] -> float [0,1]
    colors = arr['rgba'][:, :3].astype(np.float32) / 255.0
    opacities = arr['rgba'][:, 3:4].astype(np.float32) / 255.0
    
    # Quaternions: uint8 [0,255] -> float [-1,1] (w,x,y,z convention)
    quats_raw = arr['rot'].astype(np.float32)
    quats = (quats_raw / 128.0) - 1.0
    # Normalize
    qnorm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / (qnorm + 1e-8)
    
    return {
        'pts': pts,
        'colors': colors,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
    }


# =============================================================================
# Camera setup (matches gs.py's setup_camera)
# =============================================================================

def setup_camera_for_render(w, h, k, w2c, near=0.01, far=100.0, bg=[0, 0, 0]):
    """Create GaussianRasterizationSettings from camera parameters.
    
    Matches the exact setup_camera() from real_world/gs/helpers.py.
    
    Args:
        w, h: image width, height
        k: 3x3 intrinsic matrix (numpy)
        w2c: 4x4 world-to-camera matrix (numpy)
        near, far: clipping planes
        bg: background color
    """
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([
        [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
        [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
        [0.0, 0.0, 1.0, 0.0],
    ]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32).cuda(),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    return cam


# =============================================================================
# LBS (Linear Blend Skinning) for differentiable GS deformation
# =============================================================================

class DifferentiableLBS:
    """Compute LBS weights once, then apply deformations differentiably.
    
    Given control points (particles) and their motions, deform GS kernel
    positions and rotations using inverse-distance-weighted blending.
    
    The key insight: LBS weights depend only on initial geometry (bones_0
    and gs_xyz_0), so we precompute them once per episode. The actual
    deformation (applying motions) is fully differentiable w.r.t. the
    particle positions.
    """
    
    def __init__(self, k_neighbors: int = 16):
        self.k = k_neighbors
        self.weights = None      # (N_gs, N_bones) sparse-ish weights
        self.relations = None    # (N_bones, k) neighbor indices
    
    def precompute(self, bones_0: torch.Tensor, gs_xyz_0: torch.Tensor):
        """Precompute LBS weights from initial bone and GS positions.
        
        Args:
            bones_0: (N_bones, 3) initial particle positions
            gs_xyz_0: (N_gs, 3) initial Gaussian kernel positions
        """
        bones_np = bones_0.detach().cpu().numpy()
        gs_np = gs_xyz_0.detach().cpu().numpy()
        
        # KNN: for each GS kernel, find k nearest bones
        k = min(self.k, len(bones_np))
        knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(bones_np)
        distances, indices = knn.kneighbors(gs_np)
        
        # Inverse-distance weights
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Store as sparse representation: (N_gs, k) indices and weights
        self.knn_indices = torch.from_numpy(indices).long().cuda()    # (N_gs, k)
        self.knn_weights = torch.from_numpy(weights).float().cuda()   # (N_gs, k)
        
        # Bone-bone relations for rotation estimation
        k_rel = min(self.k, len(bones_np) - 1)
        knn_rel = NearestNeighbors(n_neighbors=k_rel + 1, algorithm='kd_tree').fit(bones_np)
        _, rel_indices = knn_rel.kneighbors(bones_np)
        self.relations = torch.from_numpy(rel_indices[:, 1:]).long().cuda()  # (N_bones, k)
    
    def deform(
        self,
        bones_prev: torch.Tensor,
        bones_curr: torch.Tensor,
        gs_xyz_prev: torch.Tensor,
        gs_quat_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LBS deformation — DIFFERENTIABLE w.r.t. bones_curr.
        
        Args:
            bones_prev: (N_bones, 3) previous particle positions
            bones_curr: (N_bones, 3) current (predicted) particle positions  
            gs_xyz_prev: (N_gs, 3) previous GS positions
            gs_quat_prev: (N_gs, 4) previous GS quaternions
            
        Returns:
            gs_xyz_new: (N_gs, 3) deformed GS positions (has grad)
            gs_quat_new: (N_gs, 4) deformed GS quaternions
        """
        motions = bones_curr - bones_prev  # (N_bones, 3) — carries gradient!
        
        # Per-bone rotation estimation from neighbor motions
        # For each bone, estimate local rotation from its neighborhood
        rotations = self._estimate_rotations(bones_prev, motions)  # (N_bones, 3, 3)
        
        # Weighted translation: for each GS kernel, blend bone motions
        # knn_indices: (N_gs, k), motions: (N_bones, 3)
        neighbor_motions = motions[self.knn_indices]  # (N_gs, k, 3)
        translation = (self.knn_weights.unsqueeze(-1) * neighbor_motions).sum(dim=1)  # (N_gs, 3)
        
        gs_xyz_new = gs_xyz_prev + translation
        
        # Weighted rotation blending (simplified: use translation-only for now,
        # rotation estimation adds complexity but marginal benefit for the loss signal)
        # Full rotation would require quaternion SLERP blending which is expensive.
        # The position-based rendering loss already captures most of the signal.
        gs_quat_new = gs_quat_prev  # Keep rotation fixed for v1
        
        return gs_xyz_new, gs_quat_new
    
    def _estimate_rotations(self, bones: torch.Tensor, motions: torch.Tensor) -> torch.Tensor:
        """Estimate per-bone rotation from neighborhood motions.
        
        Uses Procrustes alignment: given original neighbor offsets and
        deformed neighbor offsets, find the best rotation.
        
        Returns: (N_bones, 3, 3) rotation matrices
        """
        N = bones.shape[0]
        k = self.relations.shape[1]
        
        # Original offsets: neighbor positions relative to each bone
        neighbor_pos = bones[self.relations]  # (N, k, 3)
        offsets_orig = neighbor_pos - bones.unsqueeze(1)  # (N, k, 3)
        
        # Deformed offsets
        neighbor_motions = motions[self.relations]  # (N, k, 3)
        offsets_def = offsets_orig + neighbor_motions - motions.unsqueeze(1)  # (N, k, 3)
        
        # Procrustes: R = V @ U^T from SVD of (offsets_def^T @ offsets_orig)
        H = offsets_def.transpose(1, 2) @ offsets_orig  # (N, 3, 3)
        try:
            U, S, Vh = torch.linalg.svd(H)
            R = U @ Vh
            # Fix reflections
            det = torch.det(R)
            sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
            Vh_fixed = Vh.clone()
            Vh_fixed[:, -1, :] *= sign.squeeze(-1)
            R = U @ Vh_fixed
        except Exception:
            R = torch.eye(3, device=bones.device).unsqueeze(0).expand(N, -1, -1)
        
        return R


# =============================================================================
# Ground-truth image loader
# =============================================================================

class GTImageLoader:
    """Load ground-truth RGB images and masks for a source episode/camera.
    
    Maps from sub-episode metadata (recording index, frame range) back to
    the raw RGB images and segmentation masks on disk.
    """
    
    def __init__(
        self,
        episode_dir: Path,
        source_frame_start: int,
        camera_id: int = 1,
        image_size: Tuple[int, int] = (480, 848),  # (h, w)
        skip_frame: int = 1,
    ):
        self.camera_id = camera_id
        self.h, self.w = image_size
        self.skip_frame = skip_frame
        
        # Path to raw RGB frames — directly in the episode directory
        self.rgb_dir = episode_dir / f'camera_{camera_id}' / 'rgb'
        self.mask_dir = episode_dir / f'camera_{camera_id}' / 'mask'
        self.frame_start = source_frame_start
        
        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}\n"
                f"The observation loss requires raw RGB frames alongside the GS data."
            )
        
        self.has_masks = self.mask_dir.exists()
    
    def _find_image(self, directory: Path, frame_id: int) -> Optional[np.ndarray]:
        """Find and load an image file by frame_id, trying common naming patterns."""
        for pattern in [f'{frame_id:06d}.jpg', f'{frame_id:06d}.png',
                        f'{frame_id:04d}.jpg', f'{frame_id:04d}.png']:
            path = directory / pattern
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img
        return None
    
    def load_frame(self, rollout_step: int) -> Optional[torch.Tensor]:
        """Load RGB frame corresponding to a rollout step.
        
        Args:
            rollout_step: step index in the dynamics rollout
            
        Returns:
            (3, H, W) float tensor in [0, 1], or None if not found
        """
        frame_id = self.frame_start + rollout_step * self.skip_frame
        
        img = self._find_image(self.rgb_dir, frame_id)
        if img is None:
            return None
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).cuda()
        return img_tensor
    
    def load_mask(self, rollout_step: int) -> Optional[torch.Tensor]:
        """Load segmentation mask corresponding to a rollout step.
        
        Args:
            rollout_step: step index in the dynamics rollout
            
        Returns:
            (1, H, W) float tensor in [0, 1], or None if not found
        """
        if not self.has_masks:
            return None
        
        frame_id = self.frame_start + rollout_step * self.skip_frame
        
        mask = self._find_image(self.mask_dir, frame_id)
        if mask is None:
            return None
        
        # Convert to single-channel float
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.float32)  # binary threshold
        # (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).cuda()
        return mask_tensor


# =============================================================================
# Image-space loss functions
# =============================================================================

def ssim_loss(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute 1 - SSIM between predicted and ground truth images.
    
    Args:
        pred: (3, H, W) predicted rendered image
        gt: (3, H, W) ground truth image
        
    Returns:
        scalar loss (1 - SSIM), differentiable w.r.t. pred
    """
    # Use a simple gaussian-window SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Add batch dim
    pred = pred.unsqueeze(0)  # (1, 3, H, W)
    gt = gt.unsqueeze(0)
    
    # Gaussian window
    sigma = 1.5
    gauss = torch.tensor([
        np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ], dtype=torch.float32, device=pred.device)
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
    
    pad = window_size // 2
    
    mu_pred = F.conv2d(pred, window, padding=pad, groups=3)
    mu_gt = F.conv2d(gt, window, padding=pad, groups=3)
    
    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt
    
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=3) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=pad, groups=3) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, window, padding=pad, groups=3) - mu_pred_gt
    
    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))
    
    return 1.0 - ssim_map.mean()


def masked_image_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """Combined L1 + SSIM image loss, with mask applied ONLY to GT.
    
    The rendered image already has black background outside the Gaussians,
    so masking it would introduce artificial edge artifacts (especially
    harmful for perceptual/feature-based losses like DINOv2). We only
    mask the GT to remove background clutter, then compare directly.
    
    Args:
        pred: (3, H, W) predicted rendered image (NOT masked)
        gt: (3, H, W) ground truth image
        mask: optional (1, H, W) binary mask for region of interest
        lambda_ssim: weight for SSIM component
        
    Returns:
        scalar loss, differentiable w.r.t. pred
    """
    if mask is not None:
        # Check mask has any content
        mask_sum = mask.sum()
        if mask_sum < 1.0:
            # No valid region — return zero loss
            return pred.sum() * 0.0
        
        # Only mask GT — rendered image already has black bg from rasterizer
        gt_masked = gt * mask
        
        # L1: normalize by number of masked pixels (not total pixels)
        num_mask_pixels = mask_sum * pred.shape[0]  # multiply by channels
        l1 = (pred - gt_masked).abs().sum() / num_mask_pixels
        
        # SSIM: GT masked, pred raw
        ssim = ssim_loss(pred, gt_masked)
    else:
        l1 = F.l1_loss(pred, gt)
        ssim = ssim_loss(pred, gt)
    
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim


# =============================================================================
# DINOv2 patch feature loss
# =============================================================================

class DINOv2FeatureLoss:
    """Perceptual loss using DINOv2 patch-level features.
    
    Extracts intermediate patch token features from DINOv2-ViT and computes
    cosine distance between rendered and GT feature maps. This captures
    geometric structure and texture similarity without pixel-exact matching.
    
    The model is loaded lazily on first use and cached.
    """
    
    _instance = None  # singleton — shared across all render loss modules
    
    def __init__(self, model_name: str = 'dinov2_vits14', device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.transform = None
        self._patch_size = 14  # DINOv2 default
    
    @classmethod
    def get_instance(cls, device='cuda'):
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance
    
    def _ensure_loaded(self):
        """Lazy-load DINOv2 model on first use."""
        if self.model is not None:
            return
        
        print(f'[render_loss] Loading DINOv2 ({self.model_name})...')
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # DINOv2 normalization (ImageNet stats)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        print(f'[render_loss] DINOv2 loaded: {self.model_name}')
    
    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Resize and normalize image for DINOv2.
        
        Args:
            img: (3, H, W) float tensor in [0, 1]
            
        Returns:
            (1, 3, H', W') normalized tensor where H', W' are multiples of patch_size
        """
        _, h, w = img.shape
        # Resize to nearest multiple of patch_size (DINOv2 requirement)
        new_h = (h // self._patch_size) * self._patch_size
        new_w = (w // self._patch_size) * self._patch_size
        
        img = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w),
            mode='bilinear', align_corners=False
        )
        # Normalize with ImageNet stats
        img = (img - self._mean) / self._std
        return img
    
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract patch token features from DINOv2.
        
        Args:
            img: (3, H, W) float tensor in [0, 1]
            
        Returns:
            (num_patches, feat_dim) feature tensor
        """
        self._ensure_loaded()
        x = self._preprocess(img)
        with torch.no_grad():
            # Get intermediate features — use last layer's patch tokens
            features = self.model.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # (1, num_patches, feat_dim)
        return patch_tokens.squeeze(0)  # (num_patches, feat_dim)
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute DINOv2 patch feature loss between predicted and GT images.
        
        The predicted (rendered) image is fed raw — it already has black bg
        from the rasterizer. Only the GT is masked to remove background.
        The mask is used to weight patch contributions so we only measure
        cosine distance in cloth-covered patches.
        
        Args:
            pred: (3, H, W) rendered image (NOT masked, requires_grad)
            gt: (3, H, W) ground truth image (will be masked if mask provided)
            mask: optional (1, H, W) binary mask
            
        Returns:
            scalar loss — mean cosine distance across cloth patches
        """
        self._ensure_loaded()
        
        # Mask GT only — remove background clutter from GT
        if mask is not None:
            gt = gt * mask
        
        # For the predicted image, we need gradients to flow through
        # DINOv2 forward pass. The model params are frozen but the
        # input carries grad from the rasterizer.
        pred_input = self._preprocess(pred)
        
        # Extract features WITH gradient for pred
        pred_features = self.model.forward_features(pred_input)
        pred_tokens = pred_features['x_norm_patchtokens'].squeeze(0)  # (N, D)
        
        # Extract features WITHOUT gradient for GT
        with torch.no_grad():
            gt_input = self._preprocess(gt)
            gt_features = self.model.forward_features(gt_input)
            gt_tokens = gt_features['x_norm_patchtokens'].squeeze(0)  # (N, D)
        
        # Cosine distance per patch: 1 - cos_sim
        cos_sim = F.cosine_similarity(pred_tokens, gt_tokens, dim=-1)  # (N,)
        
        # Apply mask weighting — only count cloth-covered patches
        if mask is not None:
            _, h, w = pred.shape
            new_h = (h // self._patch_size) * self._patch_size
            new_w = (w // self._patch_size) * self._patch_size
            ph = new_h // self._patch_size
            pw = new_w // self._patch_size
            
            # Downsample mask to patch resolution
            patch_mask = F.interpolate(
                mask.unsqueeze(0).float(), size=(ph, pw), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0).reshape(-1)  # (N,)
            
            # Only count patches with >50% mask coverage
            patch_mask = (patch_mask > 0.5).float()
            
            if patch_mask.sum() < 1.0:
                return pred.sum() * 0.0  # no valid patches
            
            loss = ((1.0 - cos_sim) * patch_mask).sum() / patch_mask.sum()
        else:
            loss = (1.0 - cos_sim).mean()
        
        return loss


# =============================================================================
# Main render loss module
# =============================================================================

class RenderLossModule:
    """Manages the full render-loss pipeline for a training episode.
    
    Lifecycle per training iteration:
        1. setup_episode() — called once when a new batch is loaded
           (loads GS params, camera, precomputes LBS weights)
        2. compute_loss(x_pred, step) — called at each rollout step
           (deforms GS, renders, computes image loss)
        3. The returned loss is added to the geometry loss and backprop'd
    
    The GS model is treated as a frozen oracle — only dynamics model
    gets gradients.
    """
    
    def __init__(
        self,
        cfg,
        log_root: Path,
        device: torch.device = torch.device('cuda'),
        lambda_render: float = 0.1,
        lambda_ssim: float = 0.2,
        lambda_dino: float = 0.0,
        render_every_n_steps: int = 2,
        camera_id: int = 1,
        k_neighbors: int = 16,
        image_h: int = 480,
        image_w: int = 848,
    ):
        self.cfg = cfg
        self.log_root = log_root
        self.device = device
        self.lambda_render = lambda_render
        self.lambda_ssim = lambda_ssim
        self.lambda_dino = lambda_dino
        self.render_every_n_steps = render_every_n_steps
        self.camera_id = camera_id
        self.k_neighbors = k_neighbors
        self.image_h = image_h
        self.image_w = image_w
        
        self.lbs = DifferentiableLBS(k_neighbors=k_neighbors)
        self.active = False  # set True when episode has GS data
        
        # DINOv2 feature loss (lazy-loaded on first use)
        self.dino_loss = None
        if self.lambda_dino > 0:
            self.dino_loss = DINOv2FeatureLoss.get_instance(device=str(device))
        
        # Debug image saving
        self._debug_save_counter = 0
        self._debug_save_interval = 10000  # save debug image every 100 render calls
        
        # Cached per-episode
        self.gs_params = None
        self.cam_settings = None
        self.gt_loader = None
        self.gs_xyz_curr = None
        self.gs_quat_curr = None
        self.particles_prev = None
    
    def setup_episode(
        self,
        episode_name: str,
        particles_0: torch.Tensor,
    ) -> bool:
        """Initialize render loss for an episode.
        
        Args:
            episode_name: e.g. 'episode_0042'
            particles_0: (N, 3) initial particle positions in PGND's 
                         preprocessed coordinate space
                         
        Returns:
            True if GS data is available and render loss is active
            
        Path resolution:
            Training data lives in:     log/data/<recording>_processed/sub_episodes_v/
            GS reconstructions live in: log/data_cloth/<recording>_processed/episode_XXXX/gs/
            
            We extract the recording name from metadata, then search data_cloth/
            for the matching recording's GS data. The source_episode_id from
            meta.txt tells us which recording episode to look in.
        """
        cfg = self.cfg
        self.active = False
        
        # Resolve source data paths from metadata
        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        meta_path = source_dataset_root / episode_name / 'meta.txt'
        
        if not meta_path.exists():
            return False
        
        meta = np.loadtxt(str(meta_path))
        
        # Load metadata.json to find the source data directory
        metadata_path = source_dataset_root / 'metadata.json'
        if not metadata_path.exists():
            return False
            
        import json
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
        
        source_episode_id = int(meta[0])
        n_history = int(cfg.sim.n_history)
        load_skip = int(cfg.train.dataset_load_skip_frame)
        ds_skip = int(cfg.train.dataset_skip_frame)
        source_frame_start = int(meta[1]) + n_history * load_skip * ds_skip
        
        # Extract the recording name from the metadata path.
        # source_data_dir looks like: 'experiments/log/data/1223_cloth_gello_cali2_processed/sub_episodes_v'
        # We need the recording name: '1223_cloth_gello_cali2_processed'
        recording_name = source_data_dir.parent.name  # e.g. '1223_cloth_gello_cali2_processed'
        
        # GS data lives in data_cloth/, not data/
        # Look for: log/data_cloth/<recording_name>/episode_XXXX/gs/*.splat
        data_cloth_recording = self.log_root / 'data_cloth' / recording_name
        if not data_cloth_recording.exists():
            return False
        
        episode_dir = data_cloth_recording / f'episode_{source_episode_id:04d}'
        gs_dir = episode_dir / 'gs'
        
        if not gs_dir.exists():
            return False
        
        # Find closest splat to source_frame_start
        splat_files = sorted(gs_dir.glob('*.splat'))
        if not splat_files:
            return False
        
        frame_nums = [int(f.stem) for f in splat_files]
        closest_frame = min(frame_nums, key=lambda x: abs(x - source_frame_start))
        gs_path = gs_dir / f'{closest_frame:06d}.splat'
        
        # Load GS parameters (frozen — no grad)
        splat_data = read_splat_raw(str(gs_path))
        
        # Filter low-opacity gaussians
        opa = splat_data['opacities'][:, 0]
        valid = opa > 0.1
        
        self.gs_params = {
            'means3D': torch.from_numpy(splat_data['pts'][valid]).to(self.device),
            'rgb_colors': torch.from_numpy(splat_data['colors'][valid]).to(self.device),
            'scales': torch.from_numpy(splat_data['scales'][valid]).to(self.device),
            'quats': torch.from_numpy(splat_data['quats'][valid]).to(self.device),
            'opacities': torch.from_numpy(splat_data['opacities'][valid]).to(self.device),
        }
        
        # Transform GS positions into PGND's preprocessed coordinate space
        self._transform_gs_to_preprocessed_space(
            source_dataset_root / episode_name
        )
        
        # Load camera parameters from data_cloth/ (calibration is here)
        calib_dir = episode_dir / 'calibration'
        if not calib_dir.exists():
            return False
            
        intr = np.load(str(calib_dir / 'intrinsics.npy'))
        rvec = np.load(str(calib_dir / 'rvecs.npy'))
        tvec = np.load(str(calib_dir / 'tvecs.npy'))
        
        R = cv2.Rodrigues(rvec[self.camera_id])[0]
        t = tvec[self.camera_id, :, 0]
        
        # w2c matrix
        # rvec/tvec from calibration define world-to-camera: x_cam = R @ x_world + t
        # So c2w (camera-to-world) is: R_c2w = R.T, t_c2w = -R.T @ t
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
        
        # Setup GT image loader — RGB frames are also in data_cloth/
        try:
            self.gt_loader = GTImageLoader(
                episode_dir=data_cloth_recording / f'episode_{source_episode_id:04d}',
                source_frame_start=source_frame_start,
                camera_id=self.camera_id,
                image_size=(self.image_h, self.image_w),
                skip_frame=load_skip * ds_skip,
            )
        except FileNotFoundError:
            # RGB frames might not exist in data_cloth — that's OK, 
            # render loss just won't have GT to compare against
            return False
        
        # Precompute LBS weights
        self.lbs.precompute(
            particles_0.detach(),
            self.gs_params['means3D'],
        )
        
        # Initialize running state
        self.gs_xyz_curr = self.gs_params['means3D'].clone()
        self.gs_quat_curr = self.gs_params['quats'].clone()
        self.particles_prev = particles_0.detach().clone()
        
        self.active = True
        return True
    
    def _transform_gs_to_preprocessed_space(self, episode_data_path: Path):
        """Transform GS positions from original space to PGND preprocessed space.
        
        PGND preprocessing applies:
            1. Rotation R (y-up to z-up): x'=Rx
            2. Scale by preprocess_scale: x''= s*x'
            3. Translation to center in [0,1]^3 grid: x'''= x'' + t
        
        We apply the same transforms to the GS positions AND store the
        inverse transform so we can undo it before rendering (since the
        camera extrinsics are in the original coordinate system).
        """
        cfg = self.cfg
        dx = cfg.sim.num_grids[-1]
        
        # Load the original particle trajectory to compute the same transforms
        traj_path = episode_data_path / 'traj.npz'
        if not traj_path.exists():
            return
            
        xyz_orig = np.load(str(traj_path))['xyz']  # (T, N, 3)
        xyz = torch.tensor(xyz_orig, dtype=torch.float32)
        
        # Step 1: Rotation (same as in dataset preprocessing)
        R = torch.tensor(
            [[1, 0, 0],
             [0, 0, -1],
             [0, 1, 0]],
            dtype=torch.float32
        )
        xyz = torch.einsum('nij,jk->nik', xyz, R.T)
        
        # Step 2: Scale
        scale = cfg.sim.preprocess_scale
        xyz = xyz * scale
        
        # Step 3: Translation  
        if cfg.sim.preprocess_with_table:
            global_translation = torch.tensor([
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ], dtype=xyz.dtype)
        else:
            global_translation = torch.tensor([
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ], dtype=xyz.dtype)
        
        # Apply to GS positions (forward: original → preprocessed)
        gs_pts = self.gs_params['means3D'].cpu()
        gs_pts = gs_pts @ R
        gs_pts = gs_pts * scale
        gs_pts = gs_pts + global_translation
        self.gs_params['means3D'] = gs_pts.to(self.device)
        
        # Store inverse transform for rendering:
        # inverse: preprocessed → original
        #   x_orig = R^T * ((x_preproc - t) / s)
        self._preproc_R = R.to(self.device)
        self._preproc_scale = scale
        self._preproc_translation = global_translation.to(self.device)
    
    def compute_loss(
        self,
        x_pred: torch.Tensor,
        rollout_step: int,
    ) -> Optional[torch.Tensor]:
        """Compute render loss for a single rollout step.
        
        Args:
            x_pred: (bsz, N, 3) predicted particle positions — HAS GRADIENT
            rollout_step: current step in the rollout
            
        Returns:
            scalar loss tensor (differentiable) or None if skipping this step
        """
        if not self.active:
            return None
        
        # Only render every N steps to save compute
        if rollout_step % self.render_every_n_steps != 0:
            # Still update particles_prev for next render step
            self.particles_prev = x_pred[0].detach().clone()
            return None
        
        # Load GT image for this step
        gt_image = self.gt_loader.load_frame(rollout_step)
        if gt_image is None:
            self.particles_prev = x_pred[0].detach().clone()
            return None
        
        # Use first batch element for rendering (batch rendering is expensive)
        particles_curr = x_pred[0]  # (N, 3) — carries gradient!
        
        # LBS deformation: particles → GS kernel positions
        gs_xyz_new, gs_quat_new = self.lbs.deform(
            bones_prev=self.particles_prev,
            bones_curr=particles_curr,
            gs_xyz_prev=self.gs_xyz_curr,
            gs_quat_prev=self.gs_quat_curr,
        )
        
        # Build render data
        # IMPORTANT: gs_xyz_new is in PGND preprocessed space (for LBS).
        # The camera extrinsics expect original world coordinates.
        # Apply inverse preprocessing: x_orig = ((x_preproc - t) / s) @ R.T
        gs_xyz_render = (gs_xyz_new - self._preproc_translation) / self._preproc_scale
        gs_xyz_render = gs_xyz_render @ self._preproc_R.T  # Inverse of forward (pts @ R) is (pts @ R.T)
        
        render_data = {
            'means3D': gs_xyz_render,
            'colors_precomp': self.gs_params['rgb_colors'],
            'rotations': F.normalize(gs_quat_new, dim=-1),
            'opacities': self.gs_params['opacities'],
            'scales': self.gs_params['scales'],
            'means2D': torch.zeros_like(gs_xyz_render, requires_grad=True, device="cuda") + 0,
        }
        
        # Setup camera and render (DIFFERENTIABLE through means3D!)
        cam = setup_camera_for_render(
            w=self.cam_settings['w'],
            h=self.cam_settings['h'],
            k=self.cam_settings['k'],
            w2c=self.cam_settings['w2c'],
        )
        
        rendered_image, _, _ = GaussianRasterizer(raster_settings=cam)(**render_data)
        # rendered_image: (3, H, W) — has gradient through means3D → gs_xyz_render → gs_xyz_new → particles_curr
        
        # Load segmentation mask from dataset (cloth-only region)
        gt_mask = self.gt_loader.load_mask(rollout_step)
        # gt_mask: (1, H, W) binary mask where 1=cloth, 0=background
        
        # Resize rendered image and mask to match if needed
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
        
        # Compute image loss with dataset mask — only compare in cloth region
        loss = masked_image_loss(
            pred=rendered_image,
            gt=gt_image,
            mask=gt_mask,
            lambda_ssim=self.lambda_ssim,
        )
        
        # Add DINOv2 perceptual loss if enabled
        if self.dino_loss is not None and self.lambda_dino > 0:
            dino_l = self.dino_loss.compute_loss(
                pred=rendered_image,
                gt=gt_image,
                mask=gt_mask,
            )
            loss = (1.0 - self.lambda_dino) * loss + self.lambda_dino * dino_l
        
        # Save debug images periodically
        if self._debug_save_counter % self._debug_save_interval == 0:
            self._save_debug_images(rendered_image, gt_image, gt_mask, rollout_step)
        self._debug_save_counter += 1
        
        # Update running state (detached — these are for next step's LBS reference)
        self.gs_xyz_curr = gs_xyz_new.detach()
        self.gs_quat_curr = gs_quat_new.detach()
        self.particles_prev = particles_curr.detach().clone()
        
        return loss * self.lambda_render
    
    def _save_debug_images(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor],
        rollout_step: int,
    ):
        """Save side-by-side debug images: rendered | masked GT | diff.
        
        We compare raw rendered (already has black bg) vs masked GT.
        No masking of the rendered image.
        """
        try:
            with torch.no_grad():
                rendered_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gt_np = gt.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                
                if mask is not None:
                    mask_np = mask.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 1)
                    masked_gt_np = gt_np * mask_np
                else:
                    mask_np = np.ones_like(gt_np[:, :, :1])
                    masked_gt_np = gt_np
                
                # Difference: rendered vs masked GT (this is what the loss sees)
                diff = np.abs(rendered_np - masked_gt_np)
                diff_amplified = np.clip(diff * 5.0, 0, 1)
                
                # Side-by-side: rendered | masked GT | diff
                h, w = rendered_np.shape[:2]
                gap = 2
                canvas = np.zeros((h + 30, w * 3 + gap * 2, 3), dtype=np.float32)
                
                # Row of images
                y0 = 30  # space for labels
                canvas[y0:y0+h, 0:w] = rendered_np
                canvas[y0:y0+h, w+gap:2*w+gap] = masked_gt_np
                canvas[y0:y0+h, 2*w+2*gap:3*w+2*gap] = diff_amplified
                
                canvas_uint8 = (canvas * 255).astype(np.uint8)
                
                # Add labels with cv2
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas_uint8, 'Rendered (raw)', (10, 22), font, 0.6, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'GT (masked)', (w+gap+10, 22), font, 0.6, (255,255,255), 1)
                cv2.putText(canvas_uint8, 'Diff (5x)', (2*w+2*gap+10, 22), font, 0.6, (255,255,255), 1)
                
                out_dir = self.log_root / 'render_loss_debug'
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / f'debug_{self._debug_save_counter:06d}_step{rollout_step}.jpg'
                cv2.imwrite(str(out_path), cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR))
                
                # Also try W&B logging
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'render_debug/comparison': wandb.Image(
                                canvas_uint8,
                                caption=f'GS Rendered | GT Masked | Render Masked | Diff (step {rollout_step})'
                            ),
                        }, commit=False)
                except ImportError:
                    pass
        except Exception as e:
            print(f'[render_loss] debug image save failed: {e}')


# =============================================================================
# Integration helper — call this from train_eval.py
# =============================================================================

def create_render_loss_module(cfg, log_root, **kwargs) -> RenderLossModule:
    """Factory function to create a RenderLossModule with config defaults.
    
    Override defaults via kwargs or by adding to the PGND config:
        render_loss:
            lambda_render: 0.1
            lambda_ssim: 0.2
            render_every_n_steps: 2
            camera_id: 1
    """
    # Pull from config if available, else use defaults
    render_cfg = getattr(cfg, 'render_loss', None)
    
    defaults = {
        'lambda_render': 0.1,
        'lambda_ssim': 0.2,
        'lambda_dino': 0.0,
        'render_every_n_steps': 2,
        'camera_id': 1,
        'k_neighbors': 16,
    }
    
    if render_cfg is not None:
        for k in defaults:
            if hasattr(render_cfg, k):
                defaults[k] = getattr(render_cfg, k)
    
    defaults.update(kwargs)
    
    return RenderLossModule(cfg=cfg, log_root=log_root, **defaults)