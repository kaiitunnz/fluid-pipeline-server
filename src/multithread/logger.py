import logging

from src.logger import ILogger


class Logger(ILogger):
    """
    Logger to log server events.

    It is a thin wrapper of the `logging` library's `Logger`.
    """

    _inner: logging.Logger

    def __init__(self, inner: logging.Logger):
        """
        Parameters
        ----------
        inner : Logger
            Logger to be wrapped.
        """
        self._inner = inner

    def log(self, level: int, msg: str):
        return self._inner.log(level, msg)
