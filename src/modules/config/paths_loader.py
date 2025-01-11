import os
from dataclasses import dataclass


class PathsLoader:
    """
    A utility class to manage and return paths for known folders within the project.

    Attributes:
        BASE_DIR (str): The base directory for the project.
        PATHS_MAP (dict): A mapping of folder names to their relative paths.
    """

    @dataclass(frozen=True)
    class Paths:
        LOKI: str = "data"
        RAW_DATA: str = "data/raw"
        PROCESSED_DATA: str = "data/processed"
        OUTPUT: str = "data/output"
        SRC: str = "src"
        CONFIG_FILE: str = "src/config/config.yaml"
        LOGS: str = "src/logs"

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
    )

    PATHS_MAP = {
        Paths.LOKI: Paths.LOKI,
        Paths.OUTPUT: Paths.OUTPUT,
        Paths.RAW_DATA: Paths.RAW_DATA,
        Paths.PROCESSED_DATA: Paths.PROCESSED_DATA,
        Paths.SRC: Paths.SRC,
        Paths.CONFIG_FILE: Paths.CONFIG_FILE,
        Paths.LOGS: Paths.LOGS,
    }

    @staticmethod
    def get_folder_path(folder_name):
        """
        Get the full path to a specific folder.

        Args:
            folder_name (str): The name of the folder (must be a key in FOLDER_MAP).

        Returns:
            str: The full path to the requested folder.

        Raises:
            ValueError: If the folder name is not in the FOLDER_MAP.
        """
        if folder_name not in PathsLoader.PATHS_MAP:
            raise ValueError(
                f"Unknown folder '{folder_name}'. Available folders: {list(PathsLoader.PATHS_MAP.keys())}")

        return os.path.normpath(os.path.join(PathsLoader.BASE_DIR, PathsLoader.PATHS_MAP[folder_name]))

    @staticmethod
    def ensure_folders_exist():
        """
        Ensure all known folders exist. If not, create them.
        """
        for folder_name, relative_path in PathsLoader.PATHS_MAP.items():
            full_path = os.path.normpath(os.path.join(PathsLoader.BASE_DIR, relative_path))
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"Created folder: {full_path}")
