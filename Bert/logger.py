# logger_config.py
import logging

# Configure the logger to save logs to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/log.log',  # Log file name
    filemode='a'  # 'w' for overwriting, 'a' for appending
)

# Create a function to return the logger
def get_logger(name):
    return logging.getLogger(name)
