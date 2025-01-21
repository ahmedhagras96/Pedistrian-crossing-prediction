from modules.config.paths_loader import PATHS
from modules.features.attention_vectors.features_attention_processor import FeaturesAttentionProcessor
from modules.features.attention_vectors.features_merger import FeaturesMerger


def features_processing_pipeline():
    """
    Main function to execute feature merging and attention processing pipelines.
    """
    # Merge features across all scenarios
    FeaturesMerger.merge_all_scenarios_features(
        group_walking_dir=str(PATHS.GROUP_WALKING_FEATURES_PATH),
        speed_distance_dir=str(PATHS.SPEED_DISTANCE_FEATURES_PATH),
        output_dir=str(PATHS.MERGED_FEATURES_JSON_FILE)
    )

    # Apply attention to merged features
    FeaturesAttentionProcessor.extract_features_attentions(
        input_folder=str(PATHS.MERGED_FEATURES_JSON_FILE),
        output_folder=str(PATHS.ATTENTION_PATH),
        input_dim=5,
        output_dim=1,
        num_heads=5
    )
