import logging
import inspect
from typing import Tuple

_logger_name = "CovidIndicators"


class ModuleLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name=name)

    def get_previous_function(self) -> str:
        """
        Get the name of the function and the module from which log.debug/info/warning/error/exception/critical
        were called.
        :return: string containing the filename and the function name
        """
        frame, filename_path, line_number, function_name, lines, index = inspect.stack()[2]
        filename_components = filename_path.split("/")
        filename = filename_components[-1]
        return f"{filename} - {function_name}(): "

    def debug(self, msg, *args, **kwargs):
        msg = self.get_previous_function() + msg
        super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg = self.get_previous_function() + msg
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        msg = self.get_previous_function() + msg
        super().warning(msg, args, kwargs)

    def error(self, msg, *args, **kwargs):
        msg = self.get_previous_function() + msg
        super().error(msg, args, kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        msg = self.get_previous_function() + msg
        super().exception(msg, args, exc_info, kwargs)

    def critical(self, msg, *args, **kwargs):
        msg = self.get_previous_function() + msg
        super().critical(msg, args, kwargs)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt='%m-%d %H:%M',
    filename='./myapp.log',
    filemode='w+'
)

logging.setLoggerClass(ModuleLogger)

log = logging.getLogger(_logger_name)
