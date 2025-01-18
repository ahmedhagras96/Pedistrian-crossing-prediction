from modules.config.config_loader import ConfigLoader
from modules.config.configure_main import configure_main
from modules.config.logger import LoggerUtils
from modules.config.paths_loader import PATHS, _PathsLoader
from modules.model.training.training_pipeline import run_training

logger = None


def main():
    # run_training()
    pass


if __name__ == '__main__':
    # !Note: If any logging messages are obstructing or getting in the way, look them up and change log level from info to debug
    configure_main()
    ConfigLoader.load_config(PATHS.CONFIG_FILE)

    LoggerUtils.configure_unified_logging_file(PATHS.LOG_FILE)
    logger = LoggerUtils.get_logger(__name__)

    main()
