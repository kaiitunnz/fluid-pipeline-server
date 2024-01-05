import logging
from queue import Queue
from threading import Thread

from src.logger import ILogger


class Logger(ILogger):
    """
    Logger to be used by a process to send logging events to a listener running
    on a logging thread.
    """

    _channel: Queue

    def __init__(self, channel: Queue):
        """
        Parameters
        ----------
        channel : Queue
            Channel to be used to send logging events to the listener.
        """
        self._channel = channel

    def log(self, level: int, msg: str):
        self._channel.put((level, msg))


class LogListener:
    """
    Listener of logging events.

    It is used to create a logging thread that listens for logging events on the
    given channel.

    Attributes
    ----------
    logger : logging.Logger
        Actual logger.
    channel : Queue
        Channel on which the logging thread listens.
    name : str
        Name of the instance, used to identify itself in the server log.
    """

    logger: logging.Logger
    channel: Queue
    name: str
    _thread: Thread
    _logger: Logger

    def __init__(
        self, logger: logging.Logger, channel: Queue, name: str = "log_listener"
    ):
        """
        Parameters
        ----------
        logger : logging.Logger
            Actual logger.
        channel : Queue
            Channel on which the logging thread listens.
        name : str
            Name of the instance, used to identify itself in the server log.
        """
        self.logger = logger
        self.channel = channel
        self.name = name
        self._thread = Thread(target=self._serve, name="logger", daemon=False)
        self._logger = Logger(channel)

    def start(self):
        """Starts the logging thread"""
        self._thread.start()

    def _serve(self):
        """Serves the log listener

        It listens on the channel for logging events and logs the events with the
        logger.
        """
        try:
            while True:
                received = self.channel.get()
                if received is None:
                    break
                level, msg = received
                self.logger.log(level, msg)
        except EOFError:
            self.logger.info(f"'{self.name}' worker's channel closed.")

    def get_logger(self) -> Logger:
        """Gets the logger to be used by other processes"""
        return self._logger

    def terminate(self, _force: bool = False):
        """Stops the log listener and terminates the logging thread

        It blocks until all the pending logging events have been logged.
        """
        try:
            self.channel.put(None)
        except BrokenPipeError:
            pass
        self._thread.join()
