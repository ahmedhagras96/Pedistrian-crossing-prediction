import sys
import os
from modules.utilities.logger import LoggerUtils

main_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
log_file = os.path.normpath(os.path.abspath(os.path.join(main_path, "logs", "logs.log")))
logger = None

def main():
    ...


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    LoggerUtils.configure_unified_file_logging(log_file)
    logger = LoggerUtils.get_logger(__name__)
    main()
