import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.config import RCSLAMConfig


class Decoder(nn.Module):
    def __init__(self, config: RCSLAMConfig, is_geo_encoder=True, is_time_conditioned=False, is_imu_conditioned=True):
        super().__init__()

        self.is_time_conditioned = is_time_conditioned
        self.is_imu_conditioned = is_imu_conditioned

        if is_geo_encoder:
            mlp_hidden_dim = config.geo_mlp_hidden_dim
            mlp_bias_on = config.geo_mlp_bias_on
            mlp_level = config.geo_mlp_level
        else:
            mlp_hidden_dim = config.sem_mlp_hidden_dim
            mlp_bias_on = config.sem_mlp_bias_on
            mlp_level = config.sem_mlp_level

        input_dim = config.feature_dim
        if self.is_time_conditioned:
            input_dim += 1
        if self.is_imu_conditioned:
            input_dim += 3  # Assume imu_acc is 3D

        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_dim, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)

        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)  # SDF head
        self.nclass_out = nn.Linear(mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on)  # Semantic head
        self.normal_out = nn.Linear(mlp_hidden_dim, 3, mlp_bias_on)  # Optional: normal vector head

        self.to(config.device)

    def forward(self, feature, ts=None, imu_acc=None):
        """
        Args:
            feature: [N, F]
            ts:      [N] or None
            imu_acc: [N, 3] or None
        """
        input_feature = feature
        if self.is_time_conditioned and ts is not None:
            ts = ts.view(-1, 1)
            input_feature = torch.cat((input_feature, ts), dim=1)
        if self.is_imu_conditioned and imu_acc is not None:
            input_feature = torch.cat((input_feature, imu_acc), dim=1)

        h = input_feature
        for layer in self.layers:
            h = F.relu(layer(h))

        sdf_out = self.lout(h).squeeze(1)
        return sdf_out

    def sdf(self, feature, ts=None, imu_acc=None):
        return self.forward(feature, ts, imu_acc)

    def occupancy(self, feature, ts=None, imu_acc=None):
        out = self.sdf(feature, ts, imu_acc)
        return torch.sigmoid(out)

    def sem_label_prob(self, feature, ts=None, imu_acc=None):
        input_feature = feature
        if self.is_time_conditioned and ts is not None:
            ts = ts.view(-1, 1)
            input_feature = torch.cat((input_feature, ts), dim=1)
        if self.is_imu_conditioned and imu_acc is not None:
            input_feature = torch.cat((input_feature, imu_acc), dim=1)

        h = input_feature
        for layer in self.layers:
            h = F.relu(layer(h))

        sem_logits = self.nclass_out(h)
        return F.log_softmax(sem_logits, dim=1)

    def sem_label(self, feature, ts=None, imu_acc=None):
        return torch.argmax(self.sem_label_prob(feature, ts, imu_acc), dim=1)

    def predict_normal(self, feature, ts=None, imu_acc=None):
        input_feature = feature
        if self.is_time_conditioned and ts is not None:
            ts = ts.view(-1, 1)
            input_feature = torch.cat((input_feature, ts), dim=1)
        if self.is_imu_conditioned and imu_acc is not None:
            input_feature = torch.cat((input_feature, imu_acc), dim=1)

        h = input_feature
        for layer in self.layers:
            h = F.relu(layer(h))

        normal_vec = self.normal_out(h)
        normal_vec = F.normalize(normal_vec, dim=1)
        return normal_vec
