import os

import yaml
from modules.config.paths_loader import PATHS

class ConfigLoader:
    """
    A utility class to load and manage configuration values from a YAML file.

    Methods:
        load_config(file_path): Loads the configuration file into a static class attribute.
        get_config_value(keys): Retrieves a configuration value given a dot-separated key string.
        print_config(): Prints the entire configuration in a readable format.
        check_required_keys(required_keys): Ensures all required keys are present in the configuration.
    """
    _config = None

    @staticmethod
    def load_config(file_path: str):
        """
        Loads the configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        if ConfigLoader._config is not None:
            return

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at: {file_path}")

        with open(file_path, 'r') as f:
            try:
                ConfigLoader._config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error parsing YAML file: {e}")

    @staticmethod
    def get(keys: str):
        """
        Retrieves a configuration value given a dot-separated key string.

        Args:
            keys (str): Dot-separated key string (e.g., "paths.raw_data_path").

        Returns:
            The value associated with the provided key.

        Raises:
            KeyError: If the key does not exist in the configuration.
            RuntimeError: If the configuration is not loaded.
        """
        if ConfigLoader._config is None:
            try:
                ConfigLoader.load_config(PATHS.CONFIG_FILE)
            except:
                raise RuntimeError("Configuration is not loaded. Call 'load_config' first.")

        value = ConfigLoader._config
        for key in keys.split('.'):
            if key not in value:
                raise KeyError(f"Key '{keys}' not found in the configuration.")
            value = value[key]
        return value

    @staticmethod
    def print_config():
        """
        Prints the entire configuration in a readable format.

        Raises:
            RuntimeError: If the configuration is not loaded.
        """
        if ConfigLoader._config is None:
            raise RuntimeError("Configuration is not loaded. Call 'load_config' first.")

        print("Loaded Configuration:")
        print(yaml.dump(ConfigLoader._config, default_flow_style=False, sort_keys=False))

    @staticmethod
    def check_required_keys(required_keys: list):
        """
        Ensures all required keys are present in the configuration.

        Args:
            required_keys (list): A list of dot-separated key strings to check (e.g., ["paths.raw_data_path", "settings.batch_size"]).

        Raises:
            KeyError: If any required key is missing.
            RuntimeError: If the configuration is not loaded.
        """
        if ConfigLoader._config is None:
            raise RuntimeError("Configuration is not loaded. Call 'load_config' first.")

        missing_keys = []
        for key in required_keys:
            try:
                ConfigLoader.get(key)
            except KeyError:
                missing_keys.append(key)

        if missing_keys:
            raise KeyError(f"The following required keys are missing: {', '.join(missing_keys)}")
