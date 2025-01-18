from modules.attention.point_cloud_attention.point_cloud_attention_model import PointCloudAttentionModel


class AttentionUtils:
    def __init__(self, embed_dim, kernel_size, num_heads, logger=None):
        self.logger = logger
        self.model = PointCloudAttentionModel(embed_dim=embed_dim, kernel_size=kernel_size, num_heads=num_heads)

    def extract_3d_attention_vectors(self, batch):
        if self.logger:
            self.logger.info("Extracting 3D attention vectors")
        out, _ = self.model(batch)
        return out

    def extract_pedestrian_cloud_attention_vectors(self, batch):
        # Placeholder for actual implementation
        if self.logger:
            self.logger.info("Extracting pedestrian cloud attention vectors")
        pass

    def extract_pedestrian_features_attention_vectors(self, batch):
        # Placeholder for actual implementation
        if self.logger:
            self.logger.info("Extracting pedestrian features attention vectors")
        pass
