"""
Logging module for the Crypto Volatility Indicator.
Configures logging based on settings from the config module.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def setup_logger(config):
    """
    Set up logging based on configuration.

    Parameters:
    -----------
    config : dict or utils.config.Config
        Configuration containing logging settings

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Get logging configuration
    if hasattr(config, 'get'):
        # It's a Config object
        log_level = config.get('logging.level', 'INFO')
        log_file = config.get('logging.file', None)
        console_logging = config.get('logging.console', True)
    else:
        # It's a dictionary
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging', {}).get('file', None)
        console_logging = config.get('logging', {}).get('console', True)

    # Create logs directory if needed
    if log_file:
        log_dir = Path('logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / log_file

    # Configure root logger
    logger = logging.getLogger()

    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging initialized at level {log_level}")
    return logger


def get_logger(name):
    """
    Get a named logger.

    Parameters:
    -----------
    name : str
        Name of the logger, typically __name__

    Returns:
    --------
    logging.Logger
        Named logger
    """
    return logging.getLogger(name)


# Additional specialized loggers
def get_data_logger():
    """Get logger for data collection and processing."""
    return logging.getLogger('data')


def get_engine_logger():
    """Get logger for engine components."""
    return logging.getLogger('engine')


def get_api_logger():
    """Get logger for API interactions."""
    return logging.getLogger('api')


# Log rotation setup
def setup_log_rotation(log_file, backup_count=5):
    """
    Set up log rotation for a file.

    Parameters:
    -----------
    log_file : str or Path
        Path to log file
    backup_count : int, optional
        Number of backup files to keep
    """
    from logging.handlers import RotatingFileHandler

    root_logger = logging.getLogger()

    # Remove any existing file handlers
    for handler in [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create rotating file handler
    max_bytes = 10 * 1024 * 1024  # 10 MB
    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logging.info(f"Log rotation set up for {log_file} with {backup_count} backups")


# Performance logging
class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, name):
        """
        Initialize performance logger.

        Parameters:
        -----------
        name : str
            Name for this performance logger
        """
        self.name = name
        self.logger = logging.getLogger(f'performance.{name}')
        self.start_times = {}

    def start(self, task_name):
        """
        Start timing a task.

        Parameters:
        -----------
        task_name : str
            Name of the task to time
        """
        self.start_times[task_name] = datetime.now()

    def end(self, task_name):
        """
        End timing a task and log the duration.

        Parameters:
        -----------
        task_name : str
            Name of the task to time

        Returns:
        --------
        float
            Duration in seconds
        """
        if task_name not in self.start_times:
            self.logger.warning(f"No start time found for task: {task_name}")
            return None

        end_time = datetime.now()
        start_time = self.start_times.pop(task_name)
        duration = (end_time - start_time).total_seconds()

        self.logger.info(f"Task '{task_name}' completed in {duration:.3f} seconds")
        return duration