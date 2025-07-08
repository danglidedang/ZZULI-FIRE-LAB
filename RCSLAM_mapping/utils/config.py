import yaml
import os
import torch
from typing import List

class RCSLAMConfig:
    def __init__(self):
        self.name: str = "rc_slam"
        self.output_root: str = ""
        self.pc_path: str = ""
        self.pose_path: str = ""
        self.calib_path: str = ""
        self.label_path: str = ""
        self.load_model: bool = False
        self.model_path: str = "/"
        self.first_frame_ref: bool = True
        self.begin_frame: int = 0
        self.end_frame: int = 0
        self.every_frame: int = 1
        self.seed: int = 42
        self.num_workers: int = 12
        self.device: str = "cuda"
        self.gpu_id: str = "0"
        self.dtype = torch.float32
        self.pc_count_gpu_limit: int = 500
        self.global_shift_default: float = 0.

        self.min_range: float = 2.75
        self.pc_radius: float = 20.0
        self.min_z: float = -10.0
        self.max_z: float = 30.0
        self.rand_downsample: bool = True
        self.vox_down_m: float = 0.03
        self.rand_down_r: float = 1.0

        self.filter_noise: bool = False
        self.sor_nn: int = 25
        self.sor_std: float = 2.5
        self.estimate_normal: bool = False
        self.normal_radius_m: float = 0.2
        self.normal_max_nn: int = 20

        self.semantic_on: bool = False
        self.sem_class_count: int = 20
        self.sem_label_decimation: int = 1
        self.filter_moving_object: bool = False
        self.map_vox_down_m: float = 0.05

        self.tree_level_world: int = 10
        self.tree_level_feat: int = 4
        self.leaf_vox_size: float = 0.5
        self.feature_dim: int = 8
        self.feature_std: float = 0.05
        self.poly_int_on: bool = True
        self.octree_from_surface_samples: bool = True

        self.surface_sample_range_m: float = 0.5
        self.surface_sample_n: int = 5
        self.free_sample_begin_ratio: float = 0.3
        self.free_sample_end_dist_m: float = 0.5
        self.free_sample_n: int = 2
        self.clearance_dist_m: float = 0.3
        self.clearance_sample_n: int = 0

        self.continual_learning_reg: bool = True
        self.lambda_forget: float = 1e5
        self.cal_importance_weight_down_rate: int = 2
        self.window_replay_on: bool = True
        self.window_radius: float = 50.0
        self.occu_update_on: bool = False

        self.geo_mlp_level: int = 2
        self.geo_mlp_hidden_dim: int = 32
        self.geo_mlp_bias_on: bool = True
        self.sem_mlp_level: int = 2
        self.sem_mlp_hidden_dim: int = 32
        self.sem_mlp_bias_on: bool = True
        self.freeze_after_frame: int = 20

        self.ray_loss: bool = False
        self.main_loss_type: str = 'sdf_bce'
        self.loss_reduction: str = 'mean'
        self.sigma_sigmoid_m: float = 0.1
        self.sigma_scale_constant: float = 0.0
        self.logistic_gaussian_ratio: float = 0.55
        self.proj_correction_on: bool = False
        self.predict_sdf: bool = False
        self.neus_loss_on: bool = False
        self.loss_weight_on: bool = False
        self.behind_dropoff_on: bool = False
        self.dropoff_min_sigma: float = 1.0
        self.dropoff_max_sigma: float = 5.0
        self.normal_loss_on: bool = False
        self.weight_n: float = 0.01
        self.ekional_loss_on: bool = False
        self.weight_e: float = 0.1
        self.consistency_loss_on: bool = False
        self.weight_c: float = 1.0
        self.consistency_count: int = 1000
        self.consistency_range: float = 0.1
        self.history_weight: float = 1.0
        self.weight_s: float = 1.0
        self.time_conditioned: bool = False
        self.imu_condition_on: bool = True

        self.iters: int = 200
        self.opt_adam: bool = True
        self.bs: int = 4096
        self.lr: float = 1e-3
        self.weight_decay: float = 0
        self.adam_eps: float = 1e-15
        self.lr_level_reduce_ratio: float = 1.0
        self.lr_iters_reduce_ratio: float = 0.1
        self.lr_decay_step: List = [10000, 50000, 100000]
        self.dropout: float = 0

        self.wandb_vis_on: bool = False
        self.o3d_vis_on: bool = True
        self.eval_on: bool = False
        self.eval_outlier_thre = 0.5
        self.eval_freq_iters: int = 100
        self.vis_freq_iters: int = 100
        self.save_freq_iters: int = 100
        self.mesh_freq_frame: int = 1
        self.mc_res_m: float = 0.1
        self.pad_voxel: int = 1
        self.mc_with_octree: bool = True
        self.mc_query_level: int = 8
        self.mc_vis_level: int = 1
        self.mc_mask_on: bool = True
        self.mc_local: bool = False
        self.min_cluster_vertices: int = 50
        self.infer_bs: int = 4096
        self.occ_binary_mc: bool = False
        self.grid_loss_vis_on: bool = False
        self.mesh_vis_on: bool = True
        self.save_map: bool = False

        self.scale: float = 1.0
        self.world_size: float = 1.0

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))
        self.name = config_args["setting"]["name"]
        self.output_root = config_args["setting"]["output_root"]
        self.pc_path = config_args["setting"]["pc_path"]
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]
        if self.semantic_on:
            self.label_path = config_args["setting"]["label_path"]
        self.load_model = config_args["setting"]["load_model"]
        self.model_path = config_args["setting"]["model_path"]
        self.first_frame_ref = config_args["setting"]["first_frame_ref"]
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.every_frame = config_args["setting"]["every_frame"]
        self.device = config_args["setting"]["device"]
        self.gpu_id = config_args["setting"]["gpu_id"]
        self.min_range = config_args["process"]["min_range_m"]
        self.pc_radius = config_args["process"]["pc_radius_m"]
        self.rand_downsample = config_args["process"]["rand_downsample"]
        self.vox_down_m = config_args["process"]["vox_down_m"]
        self.rand_down_r = config_args["process"]["rand_down_r"]
        self.min_z = config_args["process"]["min_z_m"]
        self.surface_sample_range_m = config_args["sampler"]["surface_sample_range_m"]
        self.surface_sample_n = config_args["sampler"]["surface_sample_n"]
        self.free_sample_begin_ratio = config_args["sampler"]["free_sample_begin_ratio"]
        self.free_sample_end_dist_m = config_args["sampler"]["free_sample_end_dist_m"]
        self.free_sample_n = config_args["sampler"]["free_sample_n"]
        self.tree_level_world = config_args["octree"]["tree_level_world"]
        self.tree_level_feat = config_args["octree"]["tree_level_feat"]
        self.leaf_vox_size = config_args["octree"]["leaf_vox_size"]
        self.feature_dim = config_args["octree"]["feature_dim"]
        self.poly_int_on = config_args["octree"]["poly_int_on"]
        self.octree_from_surface_samples = config_args["octree"]["octree_from_surface_samples"]
        self.geo_mlp_level = config_args["decoder"]["mlp_level"]
        self.geo_mlp_hidden_dim = config_args["decoder"]["mlp_hidden_dim"]
        self.freeze_after_frame = config_args["decoder"]["freeze_after_frame"]
        self.ray_loss = config_args["loss"]["ray_loss"]
        self.main_loss_type = config_args["loss"]["main_loss_type"]
        self.sigma_sigmoid_m = config_args["loss"]["sigma_sigmoid_m"]
        self.loss_weight_on = config_args["loss"]["loss_weight_on"]
        self.behind_dropoff_on = config_args["loss"]["behind_dropoff_on"]
        self.ekional_loss_on = config_args["loss"]["ekional_loss_on"]
        self.weight_e = float(config_args["loss"]["weight_e"])
        self.continual_learning_reg = config_args["continual"]["continual_learning_reg"]
        self.lambda_forget = float(config_args["continual"]["lambda_forget"])
        self.window_replay_on = config_args["continual"]["window_replay_on"]
        self.window_radius = config_args["continual"]["window_radius_m"]
        self.iters = config_args["optimizer"]["iters"]
        self.bs = config_args["optimizer"]["batch_size"]
        self.lr = float(config_args["optimizer"]["learning_rate"])
        self.weight_decay = float(config_args["optimizer"]["weight_decay"])
        self.wandb_vis_on = config_args["eval"]["wandb_vis_on"]
        self.o3d_vis_on = config_args["eval"]["o3d_vis_on"]
        self.vis_freq_iters = config_args["eval"]["vis_freq_iters"]
        self.save_freq_iters = config_args["eval"]["save_freq_iters"]
        self.mesh_freq_frame = config_args["eval"]["mesh_freq_frame"]
        self.mc_with_octree = config_args["eval"]["mc_with_octree"]
        self.mc_res_m = config_args["eval"]["mc_res_m"]
        self.mc_vis_level = config_args["eval"]["mc_vis_level"]
        self.mc_local = config_args["eval"]["mc_local"]
        self.save_map = config_args["eval"]["save_map"]
        self.calculate_world_scale()
        self.infer_bs = self.bs * 16
        self.mc_query_level = self.tree_level_world - self.tree_level_feat + 1
        if self.window_radius <= 0:
            self.window_radius = self.pc_radius * 2.0

    def calculate_world_scale(self):
        self.world_size = self.leaf_vox_size * (2 ** (self.tree_level_world - 1))
        self.scale = 1.0 / self.world_size
