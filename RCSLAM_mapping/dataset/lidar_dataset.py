import os
import sys
import numpy as np
from numpy.linalg import inv
import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted 

from utils.config import RCSLAMConfig
from utils.pose import *
from utils.data_sampler import dataSampler
from model.feature_octree import FeatureOctree

class LiDARDataset(Dataset):
    def __init__(self, config: RCSLAMConfig, octree: FeatureOctree = None) -> None:
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)
        self.device = config.device

        # Load calibration and poses
        self.calib = read_calib_file(config.calib_path) if config.calib_path else {'Tr': np.eye(4)}
        if config.pose_path.endswith('txt'):
            self.poses_w = read_poses_file(config.pose_path, self.calib)
        elif config.pose_path.endswith('csv'):
            self.poses_w = csv_odom_to_transforms(config.pose_path)
        else:
            sys.exit("Wrong pose file format.")

        # Reference pose
        self.poses_ref = self.poses_w.copy()
        self.pc_filenames = natsorted(os.listdir(config.pc_path))
        self.total_pc_count = len(self.pc_filenames)
        self.octree = octree
        self.sampler = dataSampler(config)
        self.ray_sample_count = config.surface_sample_n + config.free_sample_n

        self.map_down_pc = o3d.geometry.PointCloud()
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()

        self.used_pc_count = 0
        self.begin_pose_inv = np.eye(4)
        for frame_id in range(self.total_pc_count):
            if frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0:
                continue
            if self.used_pc_count == 0:
                self.begin_pose_inv = inv(self.poses_w[frame_id]) if config.first_frame_ref else np.eye(4)
            self.poses_ref[frame_id] = self.begin_pose_inv @ self.poses_w[frame_id]
            self.used_pc_count += 1

        self.pool_device = "cpu" if self.used_pc_count > config.pc_count_gpu_limit and not config.continual_learning_reg and not config.window_replay_on else config.device
        self.to_cpu = self.pool_device == "cpu"
        self.sampler.dev = self.pool_device

        # Data pools
        self.coord_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sdf_label_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=self.pool_device, dtype=torch.long)
        self.weight_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.origin_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.imu_acc_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)  # NEW

    def read_point_cloud(self, filename: str):
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points, dtype=np.float64)
        else:
            sys.exit("Unsupported point cloud format")
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points)
        return pc_out

    def process_frame(self, frame_id, incremental_on=False):
        self.cur_pose_ref = self.poses_ref[frame_id]
        frame_path = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        frame_pc = self.read_point_cloud(frame_path)

        bbx = o3d.geometry.AxisAlignedBoundingBox(
            np.array([-self.config.pc_radius]*2 + [self.config.min_z]),
            np.array([self.config.pc_radius]*2 + [self.config.max_z])
        )
        frame_pc = frame_pc.crop(bbx)

        if self.config.estimate_normal:
            frame_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius_m,
                    max_nn=self.config.normal_max_nn
                )
            )

        frame_pc = frame_pc.voxel_down_sample(voxel_size=self.config.vox_down_m)
        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.pool_device)

        frame_pc.transform(self.cur_pose_ref)
        self.cur_frame_pc = copy.deepcopy(frame_pc).voxel_down_sample(voxel_size=self.config.map_vox_down_m)
        self.map_down_pc += self.cur_frame_pc
        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()

        frame_pc.scale(self.config.scale, center=(0, 0, 0))
        frame_pc_torch = torch.tensor(np.asarray(frame_pc.points), dtype=self.dtype, device=self.pool_device)
        frame_normal_torch = torch.tensor(np.asarray(frame_pc.normals), dtype=self.dtype, device=self.pool_device) if self.config.estimate_normal else None

        coord, sdf_label, normal_label, _, weight, _, _ = self.sampler.sample(
            frame_pc_torch, frame_origin_torch, frame_normal_torch, None)

        origin_repeat = frame_origin_torch.repeat(coord.shape[0], 1)
        time_repeat = torch.tensor(frame_id, dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0])
        imu_acc = torch.tensor([0.0, 0.0, 9.8], dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0], 1)

        if self.octree is not None:
            self.octree.update(coord[weight > 0].to(self.device), incremental_on)

        self.coord_pool = coord
        self.sdf_label_pool = sdf_label
        self.normal_label_pool = normal_label
        self.weight_pool = weight
        self.origin_pool = origin_repeat
        self.time_pool = time_repeat
        self.imu_acc_pool = imu_acc

    def get_batch(self):
        count = self.sdf_label_pool.shape[0]
        index = torch.randint(0, count, (self.config.bs,), device=self.pool_device)
        coord = self.coord_pool[index].to(self.device)
        sdf_label = self.sdf_label_pool[index].to(self.device)
        origin = self.origin_pool[index].to(self.device)
        ts = self.time_pool[index].to(self.device)
        normal_label = self.normal_label_pool[index].to(self.device)
        sem_label = self.sem_label_pool[index].to(self.device) if self.sem_label_pool is not None else None
        weight = self.weight_pool[index].to(self.device)
        imu_acc = self.imu_acc_pool[index].to(self.device)
        return coord, sdf_label, origin, ts, normal_label, sem_label, weight, imu_acc

    def write_merged_pc(self, out_path):
        map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out.transform(inv(self.begin_pose_inv))
        o3d.io.write_point_cloud(out_path, map_down_pc_out)
        print(f"Saved merged point cloud to {out_path}")

    def __len__(self):
        return self.sdf_label_pool.shape[0]
