import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

from utils.config import RCSLAMConfig
from utils.tools import *
from utils.loss import *
from utils.incre_learning import cal_feature_importance
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset
from utils.imu_utils import refine_normals_with_imu

def run_rcslam_mapping_incremental():

    config = RCSLAMConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("Please provide the path to the config file.\nTry: python rcslam_incre.py xxx/xxx_config.yaml")

    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path)
    dev = config.device

    octree = FeatureOctree(config)
    geo_mlp = Decoder(config, is_geo_encoder=True)
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

    geo_mlp_param = list(geo_mlp.parameters())
    sem_mlp_param = list(sem_mlp.parameters())
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0)
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    processed_frame = 0
    total_iter = 0

    if config.continual_learning_reg:
        config.loss_reduction = "sum"

    require_gradient = config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on or config.square_constraint_on

    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0):
            continue

        vis_mesh = False

        if processed_frame == config.freeze_after_frame:
            print("Freeze the decoder")
            freeze_model(geo_mlp)
            if config.semantic_on:
                freeze_model(sem_mlp)

        T0 = get_time()
        dataset.process_frame(frame_id, incremental_on=config.continual_learning_reg)

        octree_feat = list(octree.parameters())
        opt = setup_optimizer(config, octree_feat, geo_mlp_param, sem_mlp_param, sigma_size)
        octree.print_detail()
        T1 = get_time()

        for iter in tqdm(range(config.iters)):
            coord, sdf_label, origin, ts, normal_label, sem_label, weight, imu_acc = dataset.get_batch()

            if config.imu_normal_comp_on:
                normal_label = refine_normals_with_imu(normal_label, imu_acc)

            if require_gradient:
                coord.requires_grad_(True)

            feature = octree.query_feature(coord)
            sdf_pred = geo_mlp.sdf(feature)
            if config.semantic_on:
                sem_pred = sem_mlp.sem_label_prob(feature)

            surface_mask = weight > 0

            if require_gradient:
                g = get_gradient(coord, sdf_pred) * sigma_sigmoid

            if config.consistency_loss_on:
                near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count,coord.shape[0]),), device=dev)
                shift_scale = config.consistency_range * config.scale
                random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
                coord_near = coord + random_shift
                coord_near = coord_near[near_index, :]
                coord_near.requires_grad_(True)
                feature_near = octree.query_feature(coord_near)
                pred_near = geo_mlp.sdf(feature_near)
                g_near = get_gradient(coord_near, pred_near) * sigma_sigmoid

            cur_loss = 0.
            weight = torch.abs(weight)
            sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction)
            cur_loss += sdf_loss

            reg_loss = 0.
            if config.continual_learning_reg:
                reg_loss = octree.cal_regularization()
                cur_loss += config.lambda_forget * reg_loss

            eikonal_loss = 0.
            if config.ekional_loss_on:
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean()
                cur_loss += config.weight_e * eikonal_loss

            consistency_loss = 0.
            if config.consistency_loss_on:
                consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
                cur_loss += config.weight_c * consistency_loss

            normal_loss = 0.
            square_constraint_loss = 0.
            if config.normal_loss_on or config.square_constraint_on:
                g_direction = g / g.norm(2, dim=-1)

            if config.normal_loss_on:
                normal_diff = g_direction - normal_label
                normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
                cur_loss += config.weight_n * normal_loss

            if config.square_constraint_on:
                square_diff = g_direction - normal_label
                square_constraint_loss = (square_diff[surface_mask].abs()).norm(2, dim=1).mean()
                cur_loss += config.weight_square * square_constraint_loss

            sem_loss = 0.
            if config.semantic_on:
                loss_nll = nn.NLLLoss(reduction='mean')
                sem_loss = loss_nll(sem_pred[::config.sem_label_decimation,:], sem_label[::config.sem_label_decimation])
                cur_loss += config.weight_s * sem_loss

            opt.zero_grad(set_to_none=True)
            cur_loss.backward()
            opt.step()

            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {
                    'iter': total_iter,
                    'loss/total_loss': cur_loss,
                    'loss/sdf_loss': sdf_loss,
                    'loss/reg_loss': reg_loss,
                    'loss/eikonal_loss': eikonal_loss,
                    'loss/consistency_loss': consistency_loss,
                    'loss/normal_loss': normal_loss,
                    'loss/square_constraint_loss': square_constraint_loss,
                    'loss/sem_loss': sem_loss
                }
                wandb.log(wandb_log_content)

        if config.continual_learning_reg:
            opt.zero_grad(set_to_none=True)
            cal_feature_importance(dataset, octree, geo_mlp, sigma_sigmoid, config.bs, config.cal_importance_weight_down_rate, config.loss_reduction)

        T2 = get_time()

        if processed_frame == 0 or (processed_frame+1) % config.mesh_freq_frame == 0:
            print("Begin mesh reconstruction from the implicit map")
            vis_mesh = True
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id+1) + ".ply"
            map_path = run_path + '/map/sdf_map_frame_' + str(frame_id+1) + ".ply"
            if config.mc_with_octree:
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
            else:
                if config.mc_local:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.cur_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        T3 = get_time()

        if config.o3d_vis_on:
            if vis_mesh:
                cur_mesh.transform(dataset.begin_pose_inv)
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref, cur_mesh)
            else:
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref)

        if config.wandb_vis_on:
            wandb_log_content = {
                'frame': processed_frame,
                'timing(s)/preprocess': T1-T0,
                'timing(s)/mapping': T2-T1,
                'timing(s)/reconstruct': T3-T2
            }
            wandb.log(wandb_log_content)

        processed_frame += 1

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_rcslam_mapping_incremental()
