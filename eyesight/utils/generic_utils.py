import colorlog
import logging
import inspect
import shutil
import wget
import sys
import os


class Log:
    aliases = {
        logging.CRITICAL: ("critical", "crit", "c", "fatal"),
        logging.ERROR:    ("error", "err", "e"),
        logging.WARNING:  ("warning", "warn", "w"),
        logging.INFO:     ("info", "inf", "nfo", "i"),
        logging.DEBUG:    ("debug", "dbg", "d")
    }

    lvl = logging.DEBUG
    format_str = "%(log_color)s%(asctime)s | %(levelname)-8s | " \
                 "%(message)s (%(filename)s:%(lineno)d)%(reset)s"
    logging.root.setLevel(lvl)
    formatter = colorlog.ColoredFormatter(
            format_str, datefmt="%H:%M:%S", reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'reset',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            })

    stream = logging.StreamHandler()
    stream.setLevel(lvl)
    stream.setFormatter(formatter)
    logger = logging.getLogger('eyesight')
    logger.setLevel(lvl)
    logger.addHandler(stream)

    crit = c = fatal = critical = logger.critical
    err = e = error = logger.error
    warning = w = warn = logger.warning
    inf = nfo = i = info = logger.info
    dbg = d = debug = logger.debug

    @classmethod
    def _parse_level(cls, lvl):
        for log_level in cls.aliases:
            if lvl == log_level or lvl in cls.aliases[log_level]:
                return log_level
        raise TypeError("Unrecognized logging level: %s" % lvl)

    @classmethod
    def level(cls, lvl=None):
        '''Get or set the logging level.'''
        if not lvl:
            return cls._lvl
        cls._lvl = cls._parse_level(lvl)
        cls.stream.setLevel(cls._lvl)
        logging.root.setLevel(cls._lvl)


log = Log()
if 'VERBOSE' in os.environ:
    log.level(os.environ['VERBOSE'])


class Resource:
    def __init__(self, collection_name, url, is_archive=False, root_dir=None):

        # Creating, if necessary, the root directory of resources
        if root_dir is None:
            if 'EYESIGHTDIR' in os.environ:
                root_dir = os.environ['EYESIGHTDIR']
            else:
                root_dir = '~/.eyesight/'
        root_dir = os.path.expanduser(root_dir)
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # making, if necessary, the directory for this collection
        collection_dir = os.path.join(root_dir, collection_name)
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)

        # retrieving resource path
        self._path = self.retrieve(url, collection_dir)

    def path(self):
        return self._path

    @staticmethod
    def retrieve(url, folder, is_archive=False):
        file_name = os.path.basename(url)
        download_path = os.path.join(folder, file_name)

        formats = [".zip", ".tar", ".gztar", ".bztar", ".xztar"]
        if is_archive:
            for dirname in [
                    download_path.rsplit(f, 1)[0] for f in formats
                    if f in download_path]:
                if os.path.isdir(dirname):
                    return dirname
        else:
            if os.path.exists(download_path):
                return download_path

        if not os.path.exists(download_path):
            wget.download(url, download_path)

        if is_archive:
            shutil.unpack_archive(download_path, folder)
            os.remove(download_path)
            for dirname in [
                    download_path.rsplit(f, 1)[0] for f in formats
                    if f in download_path]:
                if os.path.isdir(dirname):
                    return dirname
        else:
            return download_path


def _parent_frame(skip=1):
    """Get the object of caller (parent frame)
    """
    def stack_(frame):
        framelist = []
        while frame:
            framelist.append(frame)
            frame = frame.f_back
        return framelist

    stack = stack_(sys._getframe(1))
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    return stack[start]


def caller_name(skip=1):
    """
    Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

    An empty string is returned if skipped levels exceed stack height
    """
    parentframe = _parent_frame(skip)

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


def caller_class_name(skip=1):
    """
    Get a name of a caller's class in the format module.class

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

    An empty string is returned if skipped levels exceed stack height
    """
    parentframe = _parent_frame(skip)
    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    del parentframe
    return ".".join(name)


def class_name(obj):
    return '.'.join([type(obj).__module__, type(obj).__name__])
