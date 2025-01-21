import os
from dataclasses import dataclass, fields
from pathlib import Path

__all__ = ["PATHS"]  # Only expose PATHS for external use


@dataclass(frozen=True)
class _PathsLoader:
    # Define your base directory for constructing full paths.
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    # Define all base directories
    SRC_PATH: Path = BASE_DIR / "src"
    LOGS_PATH: Path = SRC_PATH / "logs"
    CONFIG_PATH: Path = SRC_PATH / "config"
    DATA_PATH: Path = BASE_DIR / "data"
    CSV_DATA_PATH: Path = DATA_PATH / "csvs"
    RAW_DATA_PATH: Path = DATA_PATH / "raw"
    PROCESSED_DATA_PATH: Path = DATA_PATH / "processed"
    OUTPUT_DATA_PATH: Path = DATA_PATH / "output"
    SAMPLE_DATA_PATH: Path = DATA_PATH / "sample"

    # Define all processed & output directories
    FEATURES_PATH: Path = PROCESSED_DATA_PATH / "features"
    ATTENTION_PATH: Path = PROCESSED_DATA_PATH / "attention"
    PEDESTRIAN_AVATARS_PATH: Path = PROCESSED_DATA_PATH / "pedestrian_avatars"
    RECONSTRUCTED_DATA_PATH: Path = PROCESSED_DATA_PATH / "reconstruction"
    SAVED_PEDESTRIANS_PATH: Path = PROCESSED_DATA_PATH / "saved_pedestrians"
    PEDESTRIAN_FEATURES_PATH: Path = FEATURES_PATH / "pedestrian_features"
    PEDESTRIAN_AVATARS_FEATURES_PATH: Path = FEATURES_PATH / "pedestrian_avatars_features"
    GROUP_WALKING_FEATURES_PATH: Path = FEATURES_PATH / "group_walking_features"
    SPEED_DISTANCE_FEATURES_PATH: Path = FEATURES_PATH / "speed_distance"

    # Define all base files
    CONFIG_FILE: Path = CONFIG_PATH / "config.yaml"
    LOG_FILE: Path = LOGS_PATH / "log.log"
    LOKI_CSV_FILE: Path = CSV_DATA_PATH / "loki.csv"

    # Define all processed & output files
    BIN_LOKI_CSV_FILE: Path = CSV_DATA_PATH / "bin_loki.csv"
    PEDESTRIAN_POINTCLOUDS_CSV_FILE: Path = CSV_DATA_PATH / "pedestrian_pointclouds.csv"
    POINT_CLOUD_ATTENTION_JSON_FILE: Path = ATTENTION_PATH / "point_cloud_attenion.json"
    AVATAR_FILTERED_PEDESTRIANS_CSV_FILE: Path = CSV_DATA_PATH / "avatar_filtered_pedistrians.csv"
    BIN_AVATAR_FILTERED_PEDESTRIANS_CSV_FILE: Path = CSV_DATA_PATH / "bin_avatar_filtered_pedistrians.csv"
    MERGED_FEATURES_JSON_FILE: Path = FEATURES_PATH / "merged_features.json"

    BIN_FILTERED_PEDESTRIANS_CSV_PATH: Path = DATA_PATH / "b_avatar_filtered_pedistrians.csv"
    POINT_CLOUD_ATTENTION: Path = OUTPUT_DATA_PATH / "pointcloud_attention.json"

    def __post_init__(self):
        # Ensure all necessary folders exist right after initialization.
        self.__ensure_folders_exist()

    def __getattribute__(self, item):
        """
        Intercept attribute access to return the fully normalized path
        for dataclass fields, avoiding infinite recursion.
        """
        cls = type(self)
        # Check if the attribute is a defined dataclass field without triggering recursion.
        if item in cls.__dataclass_fields__:
            # Directly get the raw value without calling __getattribute__ again.
            raw_value = super().__getattribute__(item)
            full_path = os.path.normpath(os.path.join(self.BASE_DIR, raw_value))
            # print(f"Accessing {item}: {full_path}")
            return full_path

        # For non-field attributes, proceed with normal attribute access.
        return super().__getattribute__(item)

    def __ensure_folders_exist(self):
        """
        Private method to ensure that directories for each field exist.
        If a field represents a file (determined by file extension), 
        the method ensures the directory for the file exists instead of skipping it.
        """
        # Iterate over all fields defined in the dataclass.
        for field in fields(self):
            # Use super().__getattribute__ to avoid __getattribute__ override.
            relative_path = super().__getattribute__(field.name)

            # Determine the directory portion of the path.
            # If it's a file path, get its directory. Otherwise, use the path as-is.
            directory_path = (
                os.path.dirname(relative_path)
                if os.path.splitext(relative_path)[1]
                else relative_path
            )

            full_path = os.path.normpath(os.path.join(self.BASE_DIR, directory_path))
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                # print(f"Created folder: {full_path}")


# Create the singleton instance for external use.
PATHS = _PathsLoader()
