import logging
import os

from modules.config.config_loader import ConfigLoader


class ColorFormatter(logging.Formatter):
    """
    Custom formatter for console output with colored log levels.
    """
    LEVEL_COLORS = {
        logging.DEBUG: "\033[92m",  # Green
        logging.INFO: "\033[94m",  # Blue
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[1;91m"  # Bright Red
    }
    RESET_COLOR = "\033[0m"

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET_COLOR)
        message = super().format(record)
        return f"{level_color}{message}{self.RESET_COLOR}"


class LoggerUtils:
    """
    Utility class for unified logging configuration and retrieval.
    """
    _file_handler_configured = False

    @staticmethod
    def configure_unified_logging_file(log_file: str):
        """
        Configure a shared file handler for all loggers.

        Args:
            log_file (str): Path to the unified log file.
        """
        if not LoggerUtils._file_handler_configured:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            root_logger = logging.getLogger()
            file_handler = logging.FileHandler(log_file, mode='w')
            file_formatter = logging.Formatter(ConfigLoader.get("logging.format"))
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            root_logger.setLevel(ConfigLoader.get("logging.level"))
            LoggerUtils._file_handler_configured = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Retrieve a logger for the specified module or class.

        Args:
            name (str): Name of the logger.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            # color_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            color_formatter = ColorFormatter(ConfigLoader.get("logging.format"))
            console_handler.setFormatter(color_formatter)
            logger.addHandler(console_handler)
            logger.setLevel(ConfigLoader.get("logging.level"))

            logger.propagate = True
        return logger
