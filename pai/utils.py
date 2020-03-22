import os
import logging
import threading
import pathlib


class Log:
    """Monostate singleton color logger"""

    class CustomFormatter(logging.Formatter):
        """Logging Formatter to add colors and count warning / errors"""

        grey = "\x1b[38m"
        yellow = "\x1b[33m"
        red = "\x1b[31m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format_str = "%(asctime)s - %(levelname)s - " \
                "%(message)s (%(filename)s:%(lineno)d)"

        FORMATS = {
            logging.DEBUG: grey + format_str + reset,
            logging.INFO: grey + format_str + reset,
            logging.WARNING: yellow + format_str + reset,
            logging.ERROR: red + format_str + reset,
            logging.CRITICAL: bold_red + format_str + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, "%H:%M:%S")
            return formatter.format(record)

    # Configuration states
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Initializing _logger and _console_handler
    _logger = logging.getLogger('pai')
    _logger.setLevel(DEBUG)
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(DEBUG)
    _console_handler.setFormatter(CustomFormatter())
    _logger.addHandler(_console_handler)

    # primary mappings
    debug = _logger.debug
    info = _logger.info
    warning = _logger.warning
    error = _logger.error
    critical = _logger.critical

    @staticmethod
    def set_level(level):
        Log._logger.setLevel(level)
        Log._console_handler.setLevel(level)


def get_models_path():
    return os.path.join(pathlib.Path(__file__).parent, 'models')
