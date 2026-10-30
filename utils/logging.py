import logging
from pathlib import Path
from datetime import datetime

def get_logger(name=None):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name or __name__)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    log_file = log_dir / f"scat_training.log"

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
