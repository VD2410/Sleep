import logging

class LoggerConfig:
    def __init__(self, log_file='./logs/error.log', log_level=logging.ERROR):
        self.log_file = log_file
        self.log_level = log_level

    def configure_logger(self):
        logging.basicConfig(
            filename=self.log_file,
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def log_error(self, message):
        logging.error(message)