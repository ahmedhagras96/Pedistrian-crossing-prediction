from modules.attention.heads.multi_head_attention import MultiHeadAttention
from modules.features.processors.feature_loader import FeatureLoader
from modules.utilities.file_utils import FileUtils
from modules.utilities.logger import LoggerUtils


class PedestrianAttentionModel:
    """
    A utility class for applying attention mechanisms to pedestrian features.
    """

    def __init__(self, input_dim, num_heads):
        """
        Initialize the AttentionProcessor.

        Args:
            input_dim (int): Number of input feature dimensions.
            num_heads (int): Number of attention heads.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
        self.attention_layer = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)
        self.feature_loader = FeatureLoader()

    def process_scenario(self, input_file, output_file):
        """
        Apply attention to a single scenario and save the results.

        Args:
            input_file (str): Path to the input JSON file.
            output_file (str): Path to save the attention results.
        """
        try:
            self.logger.info(f"Processing attention for {input_file}")
            features = self.feature_loader.load_features(input_file)
            attention_output, attention_weights = self.attention_layer(features)

            output_data = {
                "weighted_output": attention_output.detach().cpu().numpy().tolist(),
                "attention_weights": attention_weights.detach().cpu().numpy().tolist()
            }

            FileUtils.save_json(output_data, output_file)

            self.logger.info(f"Attention results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error processing attention for {input_file}: {e}")
            raise e
