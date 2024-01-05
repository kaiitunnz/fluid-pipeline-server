import logging
from multiprocessing.queues import SimpleQueue
from threading import Thread

from src.logger import ILogger


class Logger(ILogger):
    """
    Logger to be used by a thread/process to send logging events to a listener running
    on a logging thread/process.
    """

    _channel: SimpleQueue

    def __init__(self, channel: SimpleQueue):
        """
        Parameters
        ----------
        channel : SimpleQueue
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
    channel : SimpleQueue
        Channel on which the logging thread listens.
    name : str
        Name of the instance, used to identify itself in the server log.
    """

    logger: logging.Logger
    channel: SimpleQueue
    name: str
    _thread: Thread
    _logger: Logger

    def __init__(
        self, logger: logging.Logger, channel: SimpleQueue, name: str = "log_listener"
    ):
        """
        Parameters
        ----------
        logger : logging.Logger
            Actual logger.
        channel : SimpleQueue
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
        self.logger.debug(f"[{self.name}] Started serving.")
        while True:
            received = self.channel.get()
            if received is None:
                break
            level, msg = received
            self.logger.log(level, msg)

    def get_logger(self) -> Logger:
        """Gets the logger to be used by other threads/processes"""
        return self._logger

    def terminate(self, _force: bool = False):
        """Stops the log listener and terminates the logging thread

        It blocks until all the pending logging events have been logged.
        """
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
