import torch
import torch.nn as nn
from torch_snippets.torch_loader import Report

from modules.fusion.linear_fusion_head.attention_fusion_head import AttentionFusionHead
from modules.model.training.attention_utils import AttentionUtils


class Trainer:
    def __init__(self, train_dl, val_dl, embed_dim, kernel_size, num_heads, num_epochs, logger):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.logger = logger
        self.n_epochs = num_epochs

        # Initialize utilities and models
        self.attention_utils = AttentionUtils(embed_dim, kernel_size, num_heads, logger)
        self.model = AttentionFusionHead(vector_dim=64, num_heads=4)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.reporter = Report(num_epochs)

    def train_batch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_dl:
            (batched_avatar_points, batched_reconstructed_environment,
             batched_pedestrian_features, target) = batch

            source1 = self.attention_utils.extract_3d_attention_vectors(batched_reconstructed_environment)
            source2 = self.attention_utils.extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
            source3 = self.attention_utils.extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

            outputs = self.model(source1, source2, source3)
            loss = self.criterion(outputs, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.train_dl)
        return average_loss

    @torch.no_grad()
    def val_batch(self):
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_dl:
            (batched_avatar_points, batched_reconstructed_environment,
             batched_pedestrian_features, target) = batch

            source1 = self.attention_utils.extract_3d_attention_vectors(batched_reconstructed_environment)
            source2 = self.attention_utils.extract_pedestrian_cloud_attention_vectors(batched_avatar_points)
            source3 = self.attention_utils.extract_pedestrian_features_attention_vectors(batched_pedestrian_features)

            outputs = self.model(source1, source2, source3)
            loss = self.criterion(outputs, target)
            total_loss += loss.item()

        average_loss = total_loss / len(self.val_dl)
        return average_loss

    def run_training(self):
        for epoch in range(self.n_epochs):
            for ix, _ in enumerate(self.train_dl):
                avg_loss = self.train_batch()
                self.reporter.record(epoch + (ix + 1) / len(self.train_dl), trn_loss=avg_loss, end='\r')

            for ix, _ in enumerate(self.val_dl):
                loss = self.val_batch()
                self.reporter.record(epoch + (ix + 1) / len(self.val_dl), val_loss=loss, end='\r')

            self.reporter.report_avgs(epoch + 1)
        self.reporter.plot_epochs(['trn_loss', 'val_loss'])
