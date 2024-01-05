import logging
from queue import Queue
from threading import Thread
from typing import Any, List

import src.benchmark as bench


class Benchmarker(bench.IBenchmarker):
    """
    Benchmarker to be used in worker processes.

    Attributes
    ----------
    metrics : List[str]
        List of benchmark metric names.
    """

    _channel: Queue

    def __init__(self, channel: Queue, metrics: List[str]):
        super().__init__(metrics)
        self._channel = channel

    def add(self, entry: List[Any]):
        self._channel.put(entry)


class BenchmarkListener:
    """
    Listener for benchmarking events to be used in the benchmarking thread.

    Attributes
    ----------
    benchmarker : src.benchmark.Benchmarker
        Benchmarker.
    channel : Queue
        Channel on which it listens for benchmarking events.
    logger : Logger
        Logger for logging the benchmark process to the server log.
    name : str
        Name to identify a `BenchmarkListener` object.
    """

    benchmarker: bench.Benchmarker
    channel: Queue
    logger: logging.Logger
    name: str
    _benchmarker: Benchmarker
    _thread: Thread

    def __init__(
        self,
        benchmarker: bench.Benchmarker,
        channel: Queue,
        logger: logging.Logger,
        name: str = "benchmark_listener",
    ):
        """
        Parameters
        ----------
        benchmarker : src.benchmark.Benchmarker
            Benchmarker.
        channel : Queue
            Channel on which it listens for benchmarking events.
        logger : Logger
            Logger for logging the benchmark process to the server log.
        name : str
            Name to identify a `BenchmarkListener` object.
        """

        self.benchmarker = benchmarker
        self.channel = channel
        self.logger = logger
        self.name = name
        self._benchmarker = Benchmarker(channel, self.benchmarker.metrics)
        self._thread = Thread(target=self._serve, name="logger", daemon=False)

    def start(self):
        """Starts the benchmarking thread"""
        self._thread.start()

    def _serve(self):
        """Serves the benchmark listener"""
        try:
            while True:
                entry = self.channel.get()
                if entry is None:
                    break
                self.benchmarker.add(entry)
        except EOFError:
            self.logger.info(f"'{self.name}' worker's channel closed.")

    def get_benchmarker(self):
        """Gets a benchmarker to be sent to worker processes

        Returns
        -------
        Benchmarker
            Benchmarker to be sent to worker processes.
        """
        return self._benchmarker

    def terminate(self, _force: bool = False):
        """Stops serving the benchmark listener and terminates the benchmarking thread"""
        try:
            self.channel.put(None)
        except BrokenPipeError:
            pass
        self._thread.join()
