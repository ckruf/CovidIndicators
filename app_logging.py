import logging
import inspect
import types

logger_name = "CovidIndicators"


class ModuleLogger(logging.Logger):
    def log(self, level, msg, *args, **kwargs):
        calling_function_name: str = inspect.currentframe().f_back.f_code.co_name
        caller_stack = inspect.stack()[1]
        calling_module: types.ModuleType = inspect.getmodule(caller_stack[0])
        calling_module_name = calling_module.__name__
        msg = f"{calling_module_name} + {calling_function_name}; " + msg
        super(ModuleLogger, self).log(level, msg, args, kwargs)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt='%m-%d %H:%M',
    filename='./myapp.log',
    filemode='w'
)

logging.setLoggerClass(ModuleLogger)

