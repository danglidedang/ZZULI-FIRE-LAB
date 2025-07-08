import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import wandb
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.config import RCSLAMConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset
from utils.imu_utils import refine_normals_with_imu

def run_rcslam_mapping_batch():
    config = RCSLAMConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("Please provide the path to the config file.\nTry: python rcslam_batch.py xxx/xxx_config.yaml")

    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path)
    dev = config.device

    octree = FeatureOctree(config)
    geo_mlp = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned)
    sem_mlp = Decoder(config, is_geo_encoder=False)

    if config.load_model:
        loaded_model = torch.load(config.model_path)
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        freeze_model(geo_mlp)
        if config.semantic_on:
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
            freeze_model(sem_mlp)
        if 'feature_octree' in loaded_model.keys():
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    dataset = LiDARDataset(config, octree)
    mesher = Mesher(config, octree, geo_mlp, sem_mlp)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    if config.o3d_vis_on:
        vis = MapVisualizer()

    print("Load, preprocess and sample data")
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0):
            continue
        dataset.process_frame(frame_id)

    octree_feat = list(octree.parameters())
    geo_mlp_param = list(geo_mlp.parameters())
    sem_mlp_param = list(sem_mlp.parameters())
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev) * 1.0)
    sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale

    pc_map_path = run_path + '/map/pc_map_down.ply'
    dataset.write_merged_pc(pc_map_path)

    opt = setup_optimizer(config, octree_feat, geo_mlp_param, sem_mlp_param, sigma_size)
    octree.print_detail()

    require_gradient = config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on

    print("Begin mapping")
    cur_base_lr = config.lr
    for iter in tqdm(range(config.iters)):
        T0 = get_time()
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)

        if config.ray_loss:
            coord, sample_depth, ray_depth, normal_label, sem_label, weight = dataset.get_batch()
        else:
            coord, sdf_label, origin, ts, normal_label, sem_label, weight, imu_acc = dataset.get_batch()

        if config.imu_normal_comp_on:
            normal_label = refine_normals_with_imu(normal_label, imu_acc)

        if require_gradient:
            coord.requires_grad_(True)

        T1 = get_time()
        feature = octree.query_feature(coord)
        T2 = get_time()

        if not config.time_conditioned:
            pred = geo_mlp.sdf(feature)
        else:
            pred = geo_mlp.time_conditionded_sdf(feature, ts)

        if config.semantic_on:
            sem_pred = sem_mlp.sem_label_prob(feature)

        T3 = get_time()
        surface_mask = weight > 0

        if require_gradient:
            g = get_gradient(coord, pred) * sigma_sigmoid

        if config.proj_correction_on:
            cos = torch.abs(F.cosine_similarity(g, coord - origin))
            cos[~surface_mask] = 1.0
            sdf_label = sdf_label * cos

        if config.consistency_loss_on:
            near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count, coord.shape[0]),), device=dev)
            shift_scale = config.consistency_range * config.scale
            random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            coord_near = coord + random_shift
            coord_near = coord_near[near_index, :]
            coord_near.requires_grad_(True)
            feature_near = octree.query_feature(coord_near)
            pred_near = geo_mlp.sdf(feature_near)
            g_near = get_gradient(coord_near, pred_near) * sigma_sigmoid

        cur_loss = 0.
        if config.ray_loss:
            pred_occ = torch.sigmoid(pred / sigma_size)
            pred_ray = pred_occ.reshape(config.bs, -1)
            sample_depth = sample_depth.reshape(config.bs, -1)
            if config.main_loss_type == "dr":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=False)
            elif config.main_loss_type == "dr_neus":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=True)
            cur_loss += dr_loss
        else:
            weight = torch.abs(weight)
            if config.main_loss_type == "sdf_bce":
                sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction)
            elif config.main_loss_type == "sdf_l1":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=False)
            elif config.main_loss_type == "sdf_l2":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=True)
            cur_loss += sdf_loss

        eikonal_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean()
            cur_loss += config.weight_e * eikonal_loss

        consistency_loss = 0.
        if config.consistency_loss_on:
            consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
            cur_loss += config.weight_c * consistency_loss

        normal_loss = 0.
        if config.normal_loss_on:
            g_direction = g / g.norm(2, dim=-1)
            normal_diff = g_direction - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
            cur_loss += config.weight_n * normal_loss

        square_constraint_loss = 0.
        if config.square_constraint_on:
            square_diff = g_direction - normal_label
            square_constraint_loss = (square_diff[surface_mask].abs()).norm(2, dim=1).mean()
            cur_loss += config.weight_square * square_constraint_loss

        sem_loss = 0.
        if config.semantic_on:
            loss_nll = nn.NLLLoss(reduction='mean')
            sem_loss = loss_nll(sem_pred[::config.sem_label_decimation, :], sem_label[::config.sem_label_decimation])
            cur_loss += config.weight_s * sem_loss

        T4 = get_time()
        opt.zero_grad(set_to_none=True)
        cur_loss.backward()
        opt.step()
        T5 = get_time()

        if config.wandb_vis_on:
            wandb_log_content = {
                'iter': iter,
                'loss/total_loss': cur_loss,
                'loss/sdf_loss': sdf_loss,
                'loss/eikonal_loss': eikonal_loss,
                'loss/normal_loss': normal_loss,
                'loss/consistency_loss': consistency_loss,
                'loss/square_constraint_loss': square_constraint_loss,
                'loss/sem_loss': sem_loss,
                'timing(s)/load': T1 - T0,
                'timing(s)/get_indices': T2 - T1,
                'timing(s)/inference': T3 - T2,
                'timing(s)/cal_loss': T4 - T3,
                'timing(s)/back_prop': T5 - T4,
                'timing(s)/total': T5 - T0
            }
            wandb.log(wandb_log_content)

        if (((iter+1) % config.save_freq_iters) == 0 and iter > 0):
            checkpoint_name = 'model/model_iter_' + str(iter+1)
            save_checkpoint(octree, geo_mlp, sem_mlp, opt, run_path, checkpoint_name, iter)
            save_decoder(geo_mlp, sem_mlp, run_path, checkpoint_name)

        if (((iter+1) % config.vis_freq_iters) == 0 and iter > 0):
            print("Begin mesh reconstruction from the implicit map")
            if not config.time_conditioned:
                mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + ".ply"
                map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + ".ply"
                if config.mc_with_octree:
                    cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                if config.o3d_vis_on:
                    cur_mesh.transform(dataset.begin_pose_inv)
                    vis.update_mesh(cur_mesh)
            else:
                vis.stop()
                for frame_id in tqdm(range(dataset.total_pc_count)):
                    if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % 2 != 0):
                        continue
                    mesher.ts = frame_id
                    mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
                    map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
                    if config.mc_with_octree:
                        cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                    else:
                        cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                    if config.o3d_vis_on:
                        cur_mesh.transform(dataset.begin_pose_inv)
                        vis.update_mesh(cur_mesh)

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_rcslam_mapping_batch()
