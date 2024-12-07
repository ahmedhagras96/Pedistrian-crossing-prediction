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
    def get_logger(name: str) -> logging.Logger:
        """
        Configure and return a logger with the specified name.
    
        Args:
            name (str): Name of the logger.
    
        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            handler.setFormatter(formatter)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.propagate = False
        return logger
