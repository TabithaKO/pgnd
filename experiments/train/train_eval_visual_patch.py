"""
train_eval_visual_patch.py — Integration Guide for Visual Conditioning
======================================================================

This file documents the EXACT changes needed in train_eval.py to enable
image-conditioned PGND. Copy the marked sections into your train_eval.py.

NOT a standalone script — it's a reference for modifying train_eval.py.

Changes summary:
    1. Import VisualEncoder and PGNDVisualModel
    2. In init_train(): create VisualEncoder, swap PGNDModel → PGNDVisualModel
    3. In train(): load images per episode, extract features, pass to model
    4. In eval_episode(): same image loading + feature extraction

Config flags:
    +train.use_visual=true
    +train.vis_dim=64
    +train.vis_model=dinov2_vits14
    +train.vis_camera_ids=[0,1]
"""


# =============================================================================
# CHANGE 1: Imports (add at top of train_eval.py, near other imports)
# =============================================================================

"""
### VISUAL CONDITIONING ### Imports
try:
    from train.visual_encoder import VisualEncoder
    from train.pgnd_visual import PGNDVisualModel
except ImportError:
    VisualEncoder = None
    PGNDVisualModel = None
### END VISUAL CONDITIONING ###
"""


# =============================================================================
# CHANGE 2: In init_train(), after material model creation
# =============================================================================

"""
Replace this block:
    material: nn.Module = PGNDModel(cfg)
    material.to(self.torch_device)

With:
"""

INIT_TRAIN_PATCH = '''
        ### VISUAL CONDITIONING ### Model selection
        self.use_visual = getattr(cfg.train, 'use_visual', False)
        self.vis_dim = getattr(cfg.train, 'vis_dim', 64)

        if self.use_visual and PGNDVisualModel is not None:
            print(f'[visual] Using image-conditioned PGNDVisualModel (vis_dim={self.vis_dim})')
            material: nn.Module = PGNDVisualModel(cfg, vis_dim=self.vis_dim)
            material.to(self.torch_device)

            # Load baseline weights if available (warm start)
            if cfg.model.ckpt:
                baseline_ckpt = torch.load(self.log_root / cfg.model.ckpt, map_location=self.torch_device)
                material.load_baseline_weights(baseline_ckpt['material'])
        else:
            material: nn.Module = PGNDModel(cfg)
            material.to(self.torch_device)
            self.use_visual = False
        ### END VISUAL CONDITIONING ###

        material.requires_grad_(material_requires_grad)
        material.train(True)
'''


# =============================================================================
# CHANGE 3: In init_train(), after render loss setup, create VisualEncoder
# =============================================================================

VISUAL_ENCODER_INIT = '''
        ### VISUAL CONDITIONING ### Initialize visual encoder
        self.visual_encoder = None
        if self.use_visual and VisualEncoder is not None:
            vis_camera_ids = list(getattr(cfg.train, 'vis_camera_ids', [1]))
            vis_model = getattr(cfg.train, 'vis_model', 'dinov2_vits14')
            self.visual_encoder = VisualEncoder(
                model_name=vis_model,
                feature_dim=self.vis_dim,
                camera_ids=vis_camera_ids,
                image_size=(480, 848),
                device=str(self.torch_device),
            )
            # Visual encoder projection MLP is trainable
            # DINOv2 backbone is frozen (set in VisualEncoder.__init__)
            print(f'[visual] VisualEncoder ready: {vis_model}, '
                  f'cameras={vis_camera_ids}, vis_dim={self.vis_dim}')
        ### END VISUAL CONDITIONING ###
'''


# =============================================================================
# CHANGE 4: In train(), inside the episode setup block (after render_loss setup),
#            setup visual encoder for this episode
# =============================================================================

EPISODE_VISUAL_SETUP = '''
            ### VISUAL CONDITIONING ### Setup visual encoder for this episode
            vis_feat_active = False
            if self.visual_encoder is not None and self.render_loss_module is not None:
                try:
                    # Reuse cam_settings and coord_transform from render loss module
                    rlm = self.render_loss_module
                    if hasattr(rlm, 'cam_settings') and rlm.cam_settings is not None:
                        # Build cam_settings dict for all cameras
                        cam_dict = {}
                        if hasattr(rlm, 'cam_settings_dict'):
                            cam_dict = rlm.cam_settings_dict
                        else:
                            cam_dict[getattr(cfg.train, 'render_camera_id', 1)] = rlm.cam_settings
                        self.visual_encoder.setup_episode(
                            coord_transform=rlm.coord_transform,
                            cam_settings_dict=cam_dict,
                        )
                        vis_feat_active = True
                except Exception as e:
                    print(f'[visual] Setup failed: {e}')
                    vis_feat_active = False
            ### END VISUAL CONDITIONING ###
'''


# =============================================================================
# CHANGE 5: In train(), inside the step loop, BEFORE calling self.material(),
#            extract visual features and pass them
# =============================================================================

STEP_VISUAL_FEATURES = '''
                ### VISUAL CONDITIONING ### Extract per-particle features
                vis_feat = None
                if vis_feat_active:
                    try:
                        # Load images for this step from all cameras
                        images = {}
                        for cam_id in self.visual_encoder.camera_ids:
                            img = self.render_loss_module.gt_loader.load_frame(step)
                            if img is not None:
                                images[cam_id] = img
                        if images:
                            vis_feat = self.visual_encoder(x[:, :num_particles_orig] if cfg.sim.gripper_points else x, images)
                            # vis_feat: (B, N_cloth, vis_dim)
                            if cfg.sim.gripper_points:
                                # Pad with zeros for gripper particles
                                gripper_pad = torch.zeros(
                                    vis_feat.shape[0], x.shape[1] - num_particles_orig, self.vis_dim,
                                    device=vis_feat.device, dtype=vis_feat.dtype)
                                vis_feat = torch.cat([vis_feat, gripper_pad], dim=1)
                    except Exception as e:
                        vis_feat = None
                ### END VISUAL CONDITIONING ###

                # Original model call — now with optional vis_feat
                pred = self.material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
'''


# =============================================================================
# CHANGE 6: In eval_episode(), same pattern — load images + extract features
#            Add visual encoder setup and per-step feature extraction
# =============================================================================

EVAL_VISUAL_SETUP = '''
        ### VISUAL CONDITIONING ### Setup for eval
        vis_feat_active_eval = False
        if self.visual_encoder is not None:
            try:
                render_setup = setup_episode_rendering(
                    cfg, episode, self.log_root, camera_id=1)
                if render_setup is not None:
                    cam_dict = {1: render_setup['cam_settings']}
                    self.visual_encoder.setup_episode(
                        coord_transform=render_setup['coord_transform'],
                        cam_settings_dict=cam_dict,
                    )
                    vis_feat_active_eval = True
                    _eval_gt_loader = render_setup['gt_loader']
            except Exception as e:
                print(f'[visual_eval] Setup failed: {e}')
        ### END VISUAL CONDITIONING ###
'''

EVAL_STEP_FEATURES = '''
                ### VISUAL CONDITIONING ### Per-step feature extraction
                vis_feat = None
                if vis_feat_active_eval:
                    try:
                        img = _eval_gt_loader.load_frame(step)
                        if img is not None:
                            images = {1: img}
                            x_cloth = x[:, :num_particles_orig] if cfg.sim.gripper_points else x
                            vis_feat = self.visual_encoder(x_cloth, images)
                            if cfg.sim.gripper_points:
                                gripper_pad = torch.zeros(
                                    vis_feat.shape[0], x.shape[1] - num_particles_orig, self.vis_dim,
                                    device=vis_feat.device, dtype=vis_feat.dtype)
                                vis_feat = torch.cat([vis_feat, gripper_pad], dim=1)
                    except Exception:
                        vis_feat = None
                ### END VISUAL CONDITIONING ###

                pred = self.material(x, v, x_his, v_his, enabled, vis_feat=vis_feat)
'''


# =============================================================================
# CHANGE 7: Save visual encoder weights in checkpoint
# =============================================================================

SAVE_CHECKPOINT = '''
                ### VISUAL CONDITIONING ### Save visual encoder projection weights
                if self.visual_encoder is not None:
                    ckpt_data['visual_proj'] = self.visual_encoder.proj.state_dict()
'''


# =============================================================================
# CLI FLAGS for launching visual conditioning training
# =============================================================================

CLI_EXAMPLE = '''
# Phase 1: Train image-conditioned PGND from baseline weights
cd ~/pgnd/experiments/train
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python train_eval.py \\
    'gpus=[0]' overwrite=True resume=False debug=False \\
    train.name=cloth/train_visual_v1 \\
    train.source_dataset_name=data/cloth_merged/sub_episodes_v \\
    train.dataset_name=cloth/dataset \\
    sim.preprocess_scale=0.8 sim.num_grippers=2 \\
    train.dataset_load_skip_frame=3 train.dataset_skip_frame=1 \\
    train.num_workers=0 train.batch_size=2 \\
    train.training_start_episode=162 train.training_end_episode=242 \\
    train.eval_start_episode=610 train.eval_end_episode=650 \\
    train.num_iterations=100000 \\
    train.iteration_eval_interval=5000 train.iteration_log_interval=50 \\
    train.iteration_save_interval=5000 train.resume_iteration=0 \\
    model.ckpt=cloth/train/ckpt/100000.pt \\
    +train.use_visual=true \\
    +train.vis_dim=64 \\
    +train.vis_model=dinov2_vits14 \\
    +train.vis_camera_ids=[1] \\
    +train.use_render_loss=true \\
    +train.lambda_render=0.1 \\
    +train.render_every_n_steps=1 \\
    +train.render_camera_id=1 \\
    +train.lambda_ssim=0.2 \\
    +train.use_mesh_gs=true
'''
