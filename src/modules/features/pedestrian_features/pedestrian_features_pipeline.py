from modules.config.paths_loader import PATHS
from modules.features.pedestrian_features.extractors.pedestrian_feature_extractor import PedestrianFeatureExtractor
from modules.features.pedestrian_features.extractors.scenario_processor import AllScenariosProcessor


def pedestrian_features_pipeline():
    """
    Main function to process all scenarios and extract features.
    """
    output_directory = PATHS.SPEED_DISTANCE_FEATURES_PATH
    root_directory = PATHS.SAMPLE_DATA_PATH
    fps = 5  # Set desired frames per second for processing

    processor = AllScenariosProcessor(root_directory, output_directory, fps)
    processor.process_all()

    dataset_folder = PATHS.SAMPLE_DATA_PATH
    output_folder = PATHS.GROUP_WALKING_FEATURES_PATH
    ped_avatar_folder = PATHS.PEDESTRIAN_AVATARS_FEATURES_PATH
    
    extractor = PedestrianFeatureExtractor(dataset_folder, output_folder, ped_avatar_folder, fps=5)
    extractor.extract_features()