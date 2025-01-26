import os
from pathlib import Path

from modules.config.config_loader import ConfigLoader
from modules.config.configure_main import configure_main
from modules.config.logger import Logger
from modules.config.paths_loader import PATHS

if __name__ == "__main__":
    # !Note: If any logging messages are obstructing or getting in the way, look them up and change log level from info to debug
    # pipreqs . --encoding=utf-8-sig --force
    configure_main()
    ConfigLoader.load_config(PATHS.CONFIG_FILE)
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("main.log"))
    logger = Logger.get_logger("Main")
    logger.info("Starting main script")
    
    if not os.path.exists(PATHS.LOKI_CSV_FILE):
        logger.error(f"No Loki csv file found at {PATHS.LOKI_CSV_FILE}")
    
    if os.listdir(PATHS.RAW_DATA_PATH).__len__() == 0:
        logger.error(f"No raw data found at {PATHS.RAW_DATA_PATH}")
        
    logger.info(f"LOKI CSV & Raw Data are available")
    
