import torch
import torch.nn as nn
import kaolin as kal
import numpy as np
from collections import defaultdict
import multiprocessing
from utils.config import RCSLAMConfig

def get_dict_values(dictionary, keys, default=None):
    return [dictionary.get(key, default) for key in keys]

class FeatureOctree(nn.Module):
    def __init__(self, config: RCSLAMConfig):
        super().__init__()

        self.max_level = config.tree_level_world
        self.leaf_vox_size = config.leaf_vox_size 
        self.featured_level_num = config.tree_level_feat 
        self.free_level_num = self.max_level - self.featured_level_num + 1
        self.feature_dim = config.feature_dim
        self.feature_std = config.feature_std
        self.polynomial_interpolation = config.poly_int_on
        self.device = config.device
        self.imu_condition_on = getattr(config, "imu_condition_on", True)

        self.corners_lookup_tables = []
        self.nodes_lookup_tables = []
        for l in range(self.max_level+1):
            self.corners_lookup_tables.append({})
            self.nodes_lookup_tables.append({})

        if self.featured_level_num < 1:
            raise ValueError('No level with grid features!')
        self.hier_features = nn.ParameterList([])

        self.hierarchical_indices = [] 
        self.importance_weight = []
        self.features_last_frame = []

        self.to(config.device)

    def set_zero(self):
        with torch.no_grad():
            for n in range(len(self.hier_features)):
                self.hier_features[n][-1] = torch.zeros(1,self.feature_dim)

    def forward(self, x, imu_acc=None):
        return self.query_feature(x, imu_acc)

    def get_morton(self, sample_points, level):
        points = kal.ops.spc.quantize_points(sample_points, level)
        points_morton = kal.ops.spc.points_to_morton(points)
        sample_points_with_morton = torch.hstack((sample_points, points_morton.view(-1, 1)))
        morton_set = set(points_morton.cpu().numpy())
        return sample_points_with_morton, morton_set

    def get_octree_nodes(self, level):
        nodes_morton = list(self.nodes_lookup_tables[level].keys())
        nodes_morton = torch.tensor(nodes_morton).to(self.device, torch.int64)
        nodes_spc = kal.ops.spc.morton_to_points(nodes_morton)
        nodes_spc_np = nodes_spc.cpu().numpy()
        node_size = 2**(1-level)
        nodes_coord_scaled = (nodes_spc_np * node_size) - 1. + 0.5 * node_size
        return nodes_coord_scaled

    def is_empty(self):
        return len(self.hier_features) == 0

    def clear_temp(self):
        self.hierarchical_indices = [] 
        self.importance_weight = []
        self.features_last_frame = []

    def update(self, surface_points, incremental_on=False):
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(surface_points, self.max_level) 
        pyramid = spc.pyramids[0].cpu()
        for i in range(self.max_level+1):           
            if i < self.free_level_num:
                continue
            nodes = spc.point_hierarchies[pyramid[1, i]:pyramid[1, i+1]]
            nodes_morton = kal.ops.spc.points_to_morton(nodes).cpu().numpy().tolist()
            new_nodes_index = []
            for idx in range(len(nodes_morton)):
                if nodes_morton[idx] not in self.nodes_lookup_tables[i]:
                    new_nodes_index.append(idx)
            new_nodes = nodes[new_nodes_index]
            if new_nodes.shape[0] == 0:
                continue
            corners = kal.ops.spc.points_to_corners(new_nodes).reshape(-1,3) 
            corners_unique = torch.unique(corners, dim=0)
            corners_morton = kal.ops.spc.points_to_morton(corners_unique).cpu().numpy().tolist()
            if len(self.corners_lookup_tables[i]) == 0:
                corners_dict = dict(zip(corners_morton, range(len(corners_morton))))
                self.corners_lookup_tables[i] = corners_dict
                fts = self.feature_std*torch.randn(len(corners_dict)+1, self.feature_dim, device=self.device) 
                fts[-1] = torch.zeros(1,self.feature_dim)
                self.hier_features.append(nn.Parameter(fts)) 
                if incremental_on:
                    weights = torch.zeros(len(corners_dict)+1, self.feature_dim, device=self.device) 
                    self.importance_weight.append(weights)
                    self.features_last_frame.append(fts.clone())
            else:
                pre_size = len(self.corners_lookup_tables[i])
                for m in corners_morton:
                    if m not in self.corners_lookup_tables[i]:
                        self.corners_lookup_tables[i][m] = len(self.corners_lookup_tables[i])
                new_feature_num = len(self.corners_lookup_tables[i]) - pre_size
                new_fts = self.feature_std*torch.randn(new_feature_num+1, self.feature_dim, device=self.device) 
                new_fts[-1] = torch.zeros(1,self.feature_dim)
                cur_featured_level = i-self.free_level_num
                self.hier_features[cur_featured_level] = nn.Parameter(torch.cat((self.hier_features[cur_featured_level][:-1],new_fts),0))
                if incremental_on:
                    new_weights = torch.zeros(new_feature_num+1, self.feature_dim, device=self.device)
                    self.importance_weight[cur_featured_level] = torch.cat((self.importance_weight[cur_featured_level][:-1],new_weights),0)
                    self.features_last_frame[cur_featured_level] = (self.hier_features[cur_featured_level].clone())

            corners_m = kal.ops.spc.points_to_morton(corners).cpu().numpy().tolist()
            indexes = torch.tensor([self.corners_lookup_tables[i][x] for x in corners_m]).reshape(-1,8).numpy().tolist()
            new_nodes_morton = kal.ops.spc.points_to_morton(new_nodes).cpu().numpy().tolist()
            for k in range(len(new_nodes_morton)):
                self.nodes_lookup_tables[i][new_nodes_morton[k]] = indexes[k]

    def interpolat(self, x, level, polynomial_on=True):
        coords = ((2**level)*(x*0.5+0.5)) 
        d_coords = torch.frac(coords)
        if polynomial_on:
            tx = 3*(d_coords[:,0]**2) - 2*(d_coords[:,0]**3)
            ty = 3*(d_coords[:,1]**2) - 2*(d_coords[:,1]**3)
            tz = 3*(d_coords[:,2]**2) - 2*(d_coords[:,2]**3)
        else:
            tx = d_coords[:,0]
            ty = d_coords[:,1]
            tz = d_coords[:,2]
        _1_tx = 1-tx
        _1_ty = 1-ty
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz

        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.unsqueeze(2)
        return p

    def get_indices(self, coord):
        self.hierarchical_indices = []
        for i in range(self.featured_level_num):
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level)
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist()
            features_last_row = [-1 for t in range(8)]
            indices_list = [self.nodes_lookup_tables[current_level].get(p,features_last_row) for p in points_morton] 
            indices_torch = torch.tensor(indices_list, device=self.device) 
            self.hierarchical_indices.append(indices_torch)
        return self.hierarchical_indices

    def query_feature_with_indices(self, coord, hierarchical_indices):
        sum_features = torch.zeros(coord.shape[0], self.feature_dim, device=self.device)
        for i in range(self.featured_level_num):
            current_level = self.max_level - i
            feature_level = self.featured_level_num-i-1
            coeffs = self.interpolat(coord,current_level,self.polynomial_interpolation) 
            sum_features += (self.hier_features[feature_level][hierarchical_indices[i]]*coeffs).sum(1)
        return sum_features

    def query_feature(self, coord, imu_acc=None, faster=False):
        self.set_zero()
        if faster:
            indices = self.get_indices_fast(coord)
        else:
            indices = self.get_indices(coord)
        features = self.query_feature_with_indices(coord, indices)
        if imu_acc is not None and self.imu_condition_on:
            features = torch.cat((features, imu_acc), dim=1)
        return features

    def cal_regularization(self):
        regularization = 0.
        for i in range(self.featured_level_num):
            feature_level = self.featured_level_num-i-1
            unique_indices = self.hierarchical_indices[i].flatten().unique()
            difference = self.hier_features[feature_level][unique_indices] - self.features_last_frame[feature_level][unique_indices] 
            regularization += (self.importance_weight[feature_level][unique_indices]*(difference**2)).sum()
        return regularization

    def list_duplicates(self, seq):
        dd = defaultdict(list)
        for i,item in enumerate(seq):
            dd[item].append(i)
        return [(key,locs) for key,locs in dd.items() if len(locs)>=1] 

    def get_indices_fast(self, coord):
        self.hierarchical_indices = []
        for i in range(self.featured_level_num):
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level)
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist()
            features_last_row = [-1 for t in range(8)]

            dups_in_mortons = dict(self.list_duplicates(points_morton))
            dups_indices = np.zeros((len(points_morton), 8))
            for p in dups_in_mortons.keys():
                idx = dups_in_mortons[p]
                corner_indices = self.nodes_lookup_tables[current_level].get(p,features_last_row)
                dups_indices[idx,:] = corner_indices
            indices = torch.tensor(dups_indices, device=self.device).long()
            self.hierarchical_indices.append(indices)
        return self.hierarchical_indices

    def print_detail(self):
        print("Current Octomap:")
        total_vox_count = 0
        for level in range(self.featured_level_num):
            level_vox_size = self.leaf_vox_size*(2**(self.featured_level_num-1-level))
            level_vox_count = self.hier_features[level].shape[0]
            print("%.2f m: %d voxel corners" %(level_vox_size, level_vox_count))
            total_vox_count += level_vox_count
        total_map_memory = total_vox_count * self.feature_dim * 4 / 1024 / 1024
        print("memory: %d x %d x 4 = %.3f MB" %(total_vox_count, self.feature_dim, total_map_memory)) 
        print("--------------------------------")
