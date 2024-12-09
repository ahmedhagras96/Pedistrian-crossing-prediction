import logging


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
    @staticmethod
    def get_logger(name: str, log_file: str = None) -> logging.Logger:
        """
        Configure and return a logger with the specified name.
        Optionally logs to a file as well.
    
        Args:
            name (str): Name of the logger.
            log_file (str, optional): Path to the log file. Defaults to None.
    
        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            # Console handler with color formatting
            console_handler = logging.StreamHandler()
            color_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            console_handler.setFormatter(color_formatter)
            logger.addHandler(console_handler)

            if log_file:
                # File handler without color formatting
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        return logger
