from itertools import chain
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_snippets.torch_loader import Report

from modules.attention.pedestrian_attention.pedestrian_attention_model import PedestrianAttentionPointNetModel
from modules.attention.pointcloud_attention.attention_model import PointCloudAttentionModel
from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.features.attention_vectors.multi_head_attention import MultiHeadAttention
from modules.model.fusion_head.attention_Base_fusion import AttentionFusionHead
from modules.model.loader.data_loader import get_data_loaders

# Configure logger
Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("trainer.log"))
logger = Logger.get_logger("Trainer")


class FusionTrainer:
    def __init__(self,
                 batch_size=32,
                 embed_dim=128,
                 n_epochs=1,
                 kernel_size=3,
                 num_heads=4,
                 f_input_dim=5,
                 f_num_heads=5,
                 learning_rate=0.001):
        # Hyperparameters
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.n_epochs = n_epochs
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.f_input_dim = f_input_dim
        self.f_num_heads = f_num_heads
        self.learning_rate = learning_rate

        # Create data loaders
        self.train_dl, self.val_dl, self.test_dl = get_data_loaders(
            PATHS.PEDESTRIAN_AVATARS_PATH,
            PATHS.RECONSTRUCTED_DATA_PATH,
            PATHS.PEDESTRIAN_FEATURES_PATH,
            PATHS.BIN_AVATAR_FILTERED_PEDESTRIANS_CSV_FILE,
            batch_size=self.batch_size,
            train_set_percentage=0.7,
            val_set_percentage=0.2
        )

        # Initialize models
        self.pcd_attention_model = PointCloudAttentionModel(
            embed_dim=self.embed_dim,
            kernel_size=self.kernel_size,
            num_heads=self.num_heads
        )
        self.pointnet_model = PedestrianAttentionPointNetModel(
            input_dim=3,
            output_dim=self.embed_dim
        )
        self.features_attention_model = MultiHeadAttention(
            input_dim=self.f_input_dim,
            num_heads=self.f_num_heads,
            output_dim=self.embed_dim
        )
        self.model = AttentionFusionHead(
            vector_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        # Compute class weights
        label_data = pd.read_csv(PATHS.BIN_AVATAR_FILTERED_PEDESTRIANS_CSV_FILE)
        class_counts = label_data['intended_actions'].value_counts()
        total_samples = class_counts.sum()
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count) for count in class_counts],
            dtype=torch.float
        )

        # Loss and optimizer
        self.criterion = nn.BCELoss(weight=class_weights)
        parameters = chain(
            self.model.parameters(),
            self.pcd_attention_model.parameters(),
            self.pointnet_model.parameters(),
            self.features_attention_model.parameters()
        )
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.learning_rate)

        logger.info('Number of parameters in AttentionFusionHead: %d', sum(p.numel() for p in self.model.parameters()))

        # Report for logging training progress
        self.report = Report(self.n_epochs)

    def extract_3d_attention_vectors(self, batch):
        out, _ = self.pcd_attention_model(batch)
        return out

    def extract_pedestrian_cloud_attention_vectors(self, batch):
        return self.pointnet_model(batch)

    def extract_pedestrian_features_attention_vectors(self, batch):
        out, _ = self.features_attention_model(batch)
        return out

    def train_batch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_dl:
            batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

            source1 = self.extract_3d_attention_vectors(batched_reconstructed_environment)
            source2 = self.extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
            source3 = self.extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

            outputs = self.model(source1, source2, source3)
            loss = self.criterion(outputs, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_dl)

    @torch.no_grad()
    def val_batch(self):
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_dl:
            batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch

            source1 = self.extract_3d_attention_vectors(batched_reconstructed_environment)
            source2 = self.extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
            source3 = self.extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

            outputs = self.model(source1, source2, source3)
            loss = self.criterion(outputs, target)

            total_loss += loss.item()
        return total_loss / len(self.val_dl)

    def run_training(self):
        for epoch in range(self.n_epochs):
            N_train = len(self.train_dl)
            # Training loop
            for ix, _ in enumerate(self.train_dl):
                avg_loss = self.train_batch()
                self.report.record(epoch + (ix + 1) / N_train, trn_loss=avg_loss, end='\r')

            # Validation loop
            val_loss = 0.0
            N_val = len(self.val_dl)
            for ix, _ in enumerate(self.val_dl):
                loss = self.val_batch()
                val_loss += loss
                self.report.record(epoch + (ix + 1) / N_val, val_loss=loss, end='\r')

            self.report.report_avgs(epoch + 1)
        self.report.plot_epochs(['trn_loss', 'val_loss'])


if __name__ == "__main__":
    trainer = FusionTrainer()
    trainer.run_training()
