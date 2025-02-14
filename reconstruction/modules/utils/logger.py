﻿import logging
import os


class ColorFormatter(logging.Formatter):
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


class Logger:
    _file_handler_configured = False

    @staticmethod
    def configure_unified_file_logging(log_file: str):
        """
        Configure a shared file handler for all loggers.
        Ensures the log file and its directory exist and overwrites the log file if it already exists.
        
        Args:
            log_file (str): Path to the unified log file.
        """
        if not Logger._file_handler_configured:
            # Ensure the directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Set up the file handler in overwrite mode ('w')
            root_logger = logging.getLogger()
            file_handler = logging.FileHandler(log_file, mode='w')
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.DEBUG)
            Logger._file_handler_configured = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger for a specific class or module, configured to use the shared file handler.
        
        Args:
            name (str): Name of the logger.
        
        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            # Add a console handler with color formatting
            console_handler = logging.StreamHandler()
            color_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            console_handler.setFormatter(color_formatter)
            logger.addHandler(console_handler)

            logger.setLevel(logging.DEBUG)
            # Allows messages to propagate to the root logger
            logger.propagate = True 
        return logger
