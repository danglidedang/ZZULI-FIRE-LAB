import math
from tqdm import tqdm
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset
from utils.loss import *

def cal_feature_importance(data: LiDARDataset, octree: FeatureOctree, mlp: Decoder, sigma, bs, down_rate=1, loss_reduction='mean', loss_weight_on=False):
    sample_count = data.coord_pool.shape[0]
    batch_interval = bs * down_rate
    iter_n = math.ceil(sample_count / batch_interval)
    for n in tqdm(range(iter_n)):
        head = n * batch_interval
        tail = min((n + 1) * batch_interval, sample_count)
        batch_coord = data.coord_pool[head:tail:down_rate]
        batch_label = data.sdf_label_pool[head:tail:down_rate]
        batch_normal = data.normal_label_pool[head:tail:down_rate] if data.normal_label_pool is not None else None
        batch_imu = data.imu_acc_pool[head:tail:down_rate] if data.imu_acc_pool is not None else None
        count = batch_label.shape[0]

        octree.get_indices(batch_coord)
        features = octree.query_feature(batch_coord)
        pred_sdf = mlp(features)

        sdf_loss = sdf_bce_loss(pred_sdf, batch_label, sigma, None, loss_weight_on, loss_reduction)

        if batch_normal is not None and batch_imu is not None and hasattr(data, 'refine_normals_with_imu'):
            refined_normal = data.refine_normals_with_imu(batch_normal, batch_imu)
            normal_pred = grad(pred_sdf, batch_coord, grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0]
            normal_loss = normal_consistency_loss(normal_pred, refined_normal, reduction=loss_reduction)
            loss = sdf_loss + data.config.weight_n * normal_loss
        else:
            loss = sdf_loss

        loss.backward()

        for i in range(len(octree.importance_weight)):
            octree.importance_weight[i] += octree.hier_features[i].grad.abs()
            octree.hier_features[i].grad.zero_()
            octree.importance_weight[i][-1] *= 0
