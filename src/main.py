import sys
import os
from modules.utilities.logger import LoggerUtils

main_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
log_file = os.path.normpath(os.path.abspath(os.path.join(main_path, "logs", "logs.log")))
logger = None

def main():
    ...

def set_source_path():
    # Add the src directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, "..")
    sys.path.insert(0, src_path)

if __name__ == '__main__':
    # !Note: If any logging messages are obstructing or getting in the way, look them up and change log level from info to debug
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    LoggerUtils.configure_unified_file_logging(log_file)
    logger = LoggerUtils.get_logger(__name__)
    set_source_path()
    main()
