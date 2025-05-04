import logging
import os
from pathlib import Path


class CustomLogger:
    def __init__(self):
        self.log_directory = Path("logs")
        self.log_directory.mkdir(exist_ok=True)

    def get_logger(self, module_name: str) -> logging.Logger:
        """
        Creates or returns a logger instance for the specified module
        """
        # Create logger
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        if not logger.handlers:
            # Create file handler
            log_file = self.log_directory / f"{module_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger
