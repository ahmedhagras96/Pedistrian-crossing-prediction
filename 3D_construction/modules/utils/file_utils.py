# utils/file_manager.py
import os
import glob
from typing import List
from logger import Logger

class FileUtils:
    """
    Utility class for handling file and directory operations.
    """

    # Initialize a class-level logger
    logger = Logger.get_logger(__name__)

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path (str): Path to the file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        exists = os.path.isfile(file_path)
        FileUtils.logger.debug(f"Checking if file exists: {file_path} -> {exists}")
        return exists

    @staticmethod
    def dir_exists(dir_path: str) -> bool:
        """
        Check if a directory exists.

        Args:
            dir_path (str): Path to the directory.

        Returns:
            bool: True if directory exists, False otherwise.
        """
        exists = os.path.isdir(dir_path)
        FileUtils.logger.debug(f"Checking if directory exists: {dir_path} -> {exists}")
        return exists

    @staticmethod
    def create_dir(dir_path: str):
        """
        Create a directory if it does not exist.

        Args:
            dir_path (str): Path to the directory.

        Raises:
            OSError: If the directory cannot be created.
        """
        if not FileUtils.dir_exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                FileUtils.logger.info(f"Created directory: {dir_path}")
            except OSError as e:
                FileUtils.logger.error(f"Failed to create directory {dir_path}: {e}")
                raise OSError(f"Failed to create directory {dir_path}: {e}")
        else:
            FileUtils.logger.debug(f"Directory already exists: {dir_path}")

    @staticmethod
    def list_files(dir_path: str, pattern: str = "*") -> List[str]:
        """
        List all files in a directory matching a given pattern.

        Args:
            dir_path (str): Path to the directory.
            pattern (str, optional): Glob pattern to match files. Defaults to "*".

        Returns:
            List[str]: List of file paths matching the pattern.
        """
        search_pattern = os.path.join(dir_path, pattern)
        files = glob.glob(search_pattern)
        FileUtils.logger.debug(f"Listing files in {dir_path} with pattern '{pattern}': Found {len(files)} files")
        return files
