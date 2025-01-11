from modules.utilities.logger import LoggerUtils

class BaseUtility:
    def __init__(self):
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
