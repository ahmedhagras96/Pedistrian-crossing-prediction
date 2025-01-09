import os
import sys

from modules.utilities.logger import LoggerUtils
from modules.config.paths_loader import PathsLoader
from modules.config.config_loader import ConfigLoader

def main():
    ...


def set_source_path():
    # Appends the directory of the current file to the Python module search path to ensure modules in the same directory can be imported.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Add the src directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, "..")
    sys.path.insert(0, src_path)


if __name__ == '__main__':
    # !Note: If any logging messages are obstructing or getting in the way, look them up and change log level from info to debug
    # Sets the source path to ensure that the program can locate module files correctly using sys.path.
    set_source_path()
    
    # Ensure all project folders exist
    PathsLoader.ensure_folders_exist()
    
    # Loads the config file for global usage
    ConfigLoader.load_config(PathsLoader.get_folder_path(PathsLoader.Paths.CONFIG_FILE))
    
    # Configures the logging system to log messages to a unified log file & retrieves a logger instance.
    LoggerUtils.configure_unified_logging_file(os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.LOGS), "logs.log"))
    logger = LoggerUtils.get_logger(__name__)

    main()
