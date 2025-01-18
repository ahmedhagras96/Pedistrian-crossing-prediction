from modules.config.logger import LoggerUtils
from modules.config.paths_loader import PATHS
from modules.config.config_loader import ConfigLoader

class BaseModule:
    def __init__(self):
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
        ConfigLoader.load_config(PATHS.CONFIG_FILE)
        self.config = ConfigLoader
