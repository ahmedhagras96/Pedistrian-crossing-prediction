# paths_loader.py

import os
from dataclasses import dataclass, fields

__all__ = ["PATHS"]  # Only expose PATHS for external use


@dataclass(frozen=True)
class _PathsLoader:
    # Define your base directory for constructing full paths.
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
    )

    # Define all other needed directories
    SRC: str = "src"
    LOGS: str = "src/logs"
    DATA: str = "data"
    RAW_DATA: str = "data/raw"
    PROCESSED_DATA: str = "data/processed"
    FEATURES: str = "data/output/features"
    AVATARS_DATA: str = "data/processed/pedestrian_avatars"
    RECONSTRUCTION_DATA: str = "data/processed/reconstruction"
    PEDESTRIAN_FEATURES_DATA: str = "data/processed/pedestrian_features"
    OUTPUT_DATA: str = "data/output"

    # Define all other needed files
    CONFIG_FILE: str = "src/config/config.yaml"
    LOG_FILE: str = "src/logs/log.log"
    LOKI_CSV_PATH: str = "data/loki.csv"
    BIN_LOKI_CSV_PATH: str = "data/bin_loki.csv"
    BIN_FILTERED_PEDESTRIANS_CSV_PATH: str = "data/b_avatar_filtered_pedistrians.csv"
    POINT_CLOUD_ATTENTION: str = "data/output/point_cloud_attention.json"

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
