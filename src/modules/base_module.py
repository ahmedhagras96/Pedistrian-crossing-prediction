from modules.config.paths_loader import PathsLoader
from modules.utilities.logger import LoggerUtils
from modules.config.config_loader import ConfigLoader

class BaseModule:
    def __init__(self):
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
        ConfigLoader.load_config(PathsLoader.get_folder_path(PathsLoader.Paths.CONFIG_FILE))
        self.config = ConfigLoader
