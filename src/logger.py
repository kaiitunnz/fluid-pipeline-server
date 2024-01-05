import logging
from abc import abstractmethod


class ILogger:
    """
    Logger interface.
    """

    @abstractmethod
    def log(self, level: int, msg: str):
        """Logs the message at the specified level

        Parameters
        ----------
        level : int
            Level of the logging event.
        msg : str
            Message to be logged.
        """
        raise NotImplementedError()

    def debug(self, msg: str):
        """Logs the message at the debug level

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        self.log(logging.DEBUG, msg)  # type: ignore

    def info(self, msg: str):
        """Logs the message at the info level

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        self.log(logging.INFO, msg)  # type: ignore

    def warn(self, msg: str):
        """Logs the message at the warning level

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        self.log(logging.WARNING, msg)  # type: ignore

    def error(self, msg: str):
        """Logs the message at the error level

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        self.log(logging.ERROR, msg)  # type: ignore

    def critical(self, msg: str):
        """Logs the message at the critical level

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        self.log(logging.CRITICAL, msg)  # type: ignore
