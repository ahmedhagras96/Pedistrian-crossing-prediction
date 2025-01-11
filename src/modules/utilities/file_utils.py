import json

import pandas as pd

from modules.base_module import BaseModule
from modules.utilities.logger import LoggerUtils


class FileUtils(BaseModule):
    """
    A utility class for dealing with files such as json and csv files.
    """
    logger = LoggerUtils.get_logger(__name__)

    @staticmethod
    def load_json(file_path):
        FileUtils.logger.info(f"Loading JSON from {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_path):
        FileUtils.logger.info(f"Saving JSON to {file_path}")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def read_csv(file_path):
        FileUtils.logger.info(f"Reading CSV from {file_path}")
        return pd.read_csv(file_path)
