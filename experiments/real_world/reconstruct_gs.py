from typing import Union
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import supervision as sv
import open3d as o3d
import time
import yaml
import json
from dgl.geometry import farthest_point_sampler
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from pgnd.utils import get_root
from pgnd.ffmpeg import make_video
root: Path = get_root(__file__)

from utils.pcd_utils import visualize_o3d, depth2fgpcd
from utils.env_utils import get_bounding_box
from gs.trainer import GSTrainer
from gs.convert import save_to_splat


def load_camera(episode_data_dir):
    intr = np.load(episode_data_dir / 'calibration' / 'intrinsics.npy').astype(np.float32)
    rvec = np.load(episode_data_dir / 'calibration' / 'rvecs.npy')
    tvec = np.load(episode_data_dir / 'calibration' / 'tvecs.npy')
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    T = [tvec[i, :, 0] for i in range(tvec.shape[0])]
    extrs = np.zeros((len(R), 4, 4)).astype(np.float32)
    for i in range(len(R)):
        extrs[i, :3, :3] = R[i]
        extrs[i, :3, 3] = T[i]
        extrs[i, 3, 3] = 1
    return intr, extrs


def project(points, intr, extr):
    # extr: (4, 4)
    # intr: (3, 3)
    # points: (n_points, 3)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = points @ extr.T  # (n_points, 4)
    points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
    points = points @ intr.T
    points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
    return points


def reproject(depth, intr, extr):
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xx = xx.flatten()
    yy = yy.flatten()
    points = np.stack([xx, yy, depth.flatten()], axis=1)
    
    mask = depth.flatten() > 0
    mask = np.logical_and(mask, depth.flatten() < 2)
    points = points[mask]

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    points[:, 0] = (points[:, 0] - cx) / fx * points[:, 2]
    points[:, 1] = (points[:, 1] - cy) / fy * points[:, 2]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    inv_extr = np.linalg.inv(extr)
    points = points @ inv_extr.T
    return points[:, :3]


class GSProcessor:

    def __init__(self, name, n_his_frames, device='cuda', episode_range=None):
        self.name = name
        self.n_his_frames = n_his_frames
        self.data_dir = root / "log" / "data" / name
        
        self.device = device

        if 'merged' in name:
            self.is_merged = True
            assert os.path.exists(self.data_dir)
            info = json.load(open(self.data_dir / 'sub_episodes_v' / 'info.json'))

            source_eval_list_all = []
            for source_dir, episode_list in info.items():
                if source_dir in ['n_train', 'n_eval']:
                    continue

                source_name = Path(source_dir).parent.name

                eval_episode_start, eval_episode_end = episode_list[2], episode_list[3]

                source_eval_list = []  # (episode_id, start_frame, end_frame)
                
                source_data_dir = root.parent / Path(source_dir).parent
                if not os.path.exists(root.parent / source_dir):
                    source_data_dir = Path('/data/meta-material/data') / source_name

                for eval_episode in range(eval_episode_start, eval_episode_end):
                    assert os.path.exists(source_data_dir / 'sub_episodes_v' / f'episode_{eval_episode:04d}' / 'meta.txt')
                    meta = np.loadtxt(source_data_dir / 'sub_episodes_v' / f'episode_{eval_episode:04d}' / 'meta.txt')

                    source_eval_list.append((source_data_dir, int(meta[0]), int(meta[1]), int(meta[2])))

                source_eval_list_all.append(source_eval_list)
            
            episode_range = source_eval_list_all

        else:
            self.is_merged = False
            if episode_range is None:
                n_episodes = len(glob.glob(str(self.data_dir / "episode*")))
                episode_range = np.arange(n_episodes)

        self.episodes = episode_range

        n_cameras = 4
        self.cameras = np.arange(n_cameras)

        self.max_frames = 10000
        self.H, self.W = 480, 848
        self.bbox = get_bounding_box()  # 3D bounding box of the scene

        with open(root / "real_world" / "gs" / "config" / "default.yaml", 'r') as f:
            gs_config = yaml.load(f, Loader=yaml.CLoader)

        self.gs_trainer = GSTrainer(gs_config, device=self.device)
    
    def get_gaussian(self):
        if self.is_merged:
            for source_eval_list in self.episodes:
                self.get_gaussian_single_source(source_eval_list)
        else:
            self.get_gaussian_single_source(self.episodes)

    def get_gaussian_single_source(self, episodes):
        for episode in episodes:

            if isinstance(episode, tuple):
                data_dir = episode[0]

                episode_id = episode[1]
                start_frame = episode[2]
                end_frame = episode[3]

                start_frame = start_frame + self.n_his_frames
                
                episode_data_dir = data_dir / f"episode_{episode_id:04d}"
                os.makedirs(episode_data_dir / "gs", exist_ok=True)
                intrs, extrs = load_camera(episode_data_dir)

                pcd_paths = sorted(glob.glob(str(episode_data_dir / "pcd_clean" / "*.npz")))
                n_frames = min(len(pcd_paths), self.max_frames)

                assert start_frame < n_frames, f"episode {episode}"
                assert end_frame <= n_frames
                
            else:
                episode = int(episode)
                data_dir = self.data_dir
                episode_id = episode
                start_frame = 0

                episode_data_dir = data_dir / f"episode_{episode_id:04d}"
                os.makedirs(episode_data_dir / "gs", exist_ok=True)
                intrs, extrs = load_camera(episode_data_dir)

                pcd_paths = sorted(glob.glob(str(episode_data_dir / "pcd_clean" / "*.npz")))
                n_frames = min(len(pcd_paths), self.max_frames)

                end_frame = n_frames

            pivot_skip = 120

            for frame_id in range(start_frame, end_frame, pivot_skip):
                print(f'[get_gaussian] processing episode {episode_id}, frame {frame_id}')
                
                if os.path.exists(os.path.join(episode_data_dir / 'gs' / f'{frame_id:06d}.splat')):
                    print(f'[get_gaussian] already processed, skipping')
                    continue

                pcd_npz = np.load(episode_data_dir / "pcd_clean" / f"{frame_id:06d}.npz")

                pts = pcd_npz['pts']
                colors = pcd_npz['colors']

                # downsample
                n_points = 3000
                particle_tensor = torch.from_numpy(pts).float().cuda()[None]
                fps_idx_tensor = farthest_point_sampler(particle_tensor, n_points, start_idx=0)[0]
                pts = pts[fps_idx_tensor.cpu().numpy()]
                colors = colors[fps_idx_tensor.cpu().numpy()]
                if colors.max() > 1:
                    colors = colors.astype(np.float32) / 255

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                imgs = []
                masks = []
                R_list = []
                t_list = []
                intr_list = []
                for camera in self.cameras:
                    rgb_path = str(episode_data_dir / f'camera_{camera}' / 'rgb' / f'{frame_id:06d}.jpg')
                    mask_path = str(episode_data_dir / f'camera_{camera}' / 'mask' / f'{frame_id:06d}.png')
                    img = cv2.imread(rgb_path)  # bgr
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    mask = mask > 0  # binary mask
                    imgs.append(img)
                    masks.append(mask)
                    w2c = extrs[camera]
                    c2w = np.linalg.inv(w2c)
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    R_list.append(R)
                    t_list.append(t)
                    intr_list.append(intrs[camera])

                # save
                debug_vis = False
                if debug_vis:
                    for i in range(len(imgs)):
                        # project points
                        points = project(pts, intrs[i], extrs[i])
                        points = points.astype(np.int32)
                        img = imgs[i].copy()
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        for j in range(points.shape[0]):
                            if points[j, 0] < 0 or points[j, 0] >= self.W or points[j, 1] < 0 or points[j, 1] >= self.H:
                                continue
                            if not masks[i][points[j, 1], points[j, 0]]:
                                continue
                            cv2.circle(img, (points[j, 0], points[j, 1]), 2, (255, 0, 0), -1)
                        
                        cv2.imwrite(f"{frame_id:06d}_{i}_proj.png", img)
                        cv2.imwrite(f"{frame_id:06d}_{i}_rgb.png", cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{frame_id:06d}_{i}_mask.png", (masks[i] * 1).astype(np.uint8))

                self.gs_trainer.clear(clear_params=True)
                self.gs_trainer.update_state_no_env(pcd, imgs, masks, R_list, t_list, intr_list, n_cameras=4)
            
                os.makedirs(root / "log/gs/train", exist_ok=True)
                self.gs_trainer.train(vis_dir=str(root / "log/gs/train"))

                self.gs_dir = os.path.join(episode_data_dir / "gs" / f'{frame_id:06d}.splat')
                save_to_splat(
                    pts=self.gs_trainer.params['means3D'].detach().cpu().numpy(),
                    colors=self.gs_trainer.params['rgb_colors'].detach().cpu().numpy(),
                    scales=torch.exp(self.gs_trainer.params['log_scales']).detach().cpu().numpy(),
                    quats=torch.nn.functional.normalize(self.gs_trainer.params['unnorm_rotations'], dim=-1).detach().cpu().numpy(),
                    opacities=torch.sigmoid(self.gs_trainer.params['logit_opacities']).detach().cpu().numpy(),
                    output_file=self.gs_dir,
                    center=False,  # do not center the points
                    rotate=False,  # do not rotate the points to swap z and y
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='rope')
    parser.add_argument('--n_his_frames', type=int, default=6)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.device}"

    pp = GSProcessor(args.task + '_merged', args.n_his_frames, device)
    pp.get_gaussian()
