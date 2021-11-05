"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import sys
import logging

log_lv = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

class ColorFormatter(logging.Formatter):
    color_dic = {
        'DEBUG'   : 37,  # white
        'INFO'    : 36,  # cyan
        'WARNING' : 33,  # yellow
        'ERROR'   : 31,  # red
        'CRITICAL': 41,  # white on red bg
    }

    def format(self, record):
        color = self.color_dic.get(record.levelname, 37) # default white
        record.levelname = "\033[{}m{}\033[0m".format(color, record.levelname)
        return logging.Formatter.format(self, record)


class Logger(logging.Logger):
    NAME = 'SingletonLogger'

    @classmethod
    def get(cls, file_path=None, level='info', colorize=True):
        logging.setLoggerClass(cls)
        logger = logging.getLogger(cls.NAME)
        logging.setLoggerClass(logging.Logger) # restore
        logger.setLevel(log_lv[level])

        if logger.hasHandlers():
            # If logger already got all handlers (# handlers == 2), use the logger.
            # else, re-set handlers.
            if len(logger.handlers) == 2:
                return logger

            logger.handlers.clear()

        log_format = '%(levelname)s::%(asctime)s | %(message)s'
        date_format = '%m/%d %H:%M:%S'
        if colorize:
            formatter = ColorFormatter(log_format, date_format)
        else:
            formatter = logging.Formatter(log_format, date_format)

        # standard output handler
        # NOTE as default, StreamHandler use stderr stream instead of stdout stream.
        # Use StreamHandler(sys.stdout) for stdout stream.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file_path:
            # file output handler
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

        return logger

    def nofmt(self, msg, *args, level='info', **kwargs):
        level = log_lv[level]
        formatters = self.remove_formats()
        super().log(level, msg, *args, **kwargs)
        self.set_formats(formatters)

    def remove_formats(self):
        """ Remove all formats from logger """
        formatters = []
        for handler in self.handlers:
            formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        return formatters

    def set_formats(self, formatters):
        """ Set formats to every handler of logger """
        for handler, formatter in zip(self.handlers, formatters):
            handler.setFormatter(formatter)

    def set_file_handler(self, file_path):
        file_handler = logging.FileHandler(file_path)
        formatter = self.handlers[0].formatter
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
