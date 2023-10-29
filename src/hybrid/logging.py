import logging
from multiprocessing.queues import SimpleQueue
from threading import Thread


class Logger:
    _channel: SimpleQueue

    def __init__(self, channel: SimpleQueue):
        self._channel = channel

    def log(self, level: int, msg: str):
        self._channel.put((level, msg))

    def debug(self, msg: str):
        self.log(logging.DEBUG, msg)

    def info(self, msg: str):
        self.log(logging.INFO, msg)

    def warn(self, msg: str):
        self.log(logging.WARNING, msg)

    def error(self, msg: str):
        self.log(logging.ERROR, msg)

    def critical(self, msg: str):
        self.log(logging.CRITICAL, msg)


class LogListener:
    logger: logging.Logger
    channel: SimpleQueue
    name: str
    _thread: Thread
    _logger: Logger

    def __init__(
        self, logger: logging.Logger, channel: SimpleQueue, name: str = "log_listener"
    ):
        self.logger = logger
        self.channel = channel
        self.name = name
        self._thread = Thread(target=self._serve, name="logger", daemon=False)
        self._logger = Logger(channel)

    def start(self):
        self._thread.start()

    def _serve(self):
        self.logger.debug(f"[{self.name}] Started serving.")
        while True:
            received = self.channel.get()
            if received is None:
                break
            level, msg = received
            self.logger.log(level, msg)

    def get_logger(self):
        return self._logger

    def terminate(self, _force: bool = False):
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
