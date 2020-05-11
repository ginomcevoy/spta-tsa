import logging


class LoggerMixin(object):
    '''
    Mixin for adding logging capabilities to a class.
    Only works on instance methods.
    '''

    @property
    def logger(self):
        name = '.'.join([
            self.__module__,
            self.__class__.__name__
        ])
        return logging.getLogger(name)


def logger_for_me(func):
    '''
    Return a logger for the supplied function.
    '''
    name = '.'.join([
        func.__module__,
        func.__name__
    ])
    return logging.getLogger(name)


def setup_log(log_level_str):
    '''
    Configure the logger given a log string, e.g. 'INFO' or 'DEBUG'.
    '''
    log_level = getattr(logging, log_level_str)
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # disable library logging
    logging.getLogger('matplotlib.backends.backend_ps').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True


def setup_log_argparse(args, default_level='DEBUG'):
    '''
    Configure the logger when using argparse. Defaults to 'DEBUG'.
    '''
    log_level_str = default_level
    if hasattr(args, 'log') and args.log:
        log_level_str = args.log

    elif hasattr(args, 'logger') and args.logger:
        log_level_str = args.logger

    setup_log(log_level_str)
