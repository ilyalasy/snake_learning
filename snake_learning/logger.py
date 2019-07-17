import logging

LOG_PATH = "./logs"
FILE_NAME = "learning"

def get_logger():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, FILE_NAME))
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger("SnakeLogger")
 
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
