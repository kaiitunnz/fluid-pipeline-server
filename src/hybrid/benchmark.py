from multiprocessing.queues import SimpleQueue
from threading import Thread
from typing import Any, List

import src.benchmark as bench
from src.logger import DefaultLogger


class Benchmarker(bench.IBenchmarker):
    """
    Benchmarker to be used in worker threads/processes.

    Attributes
    ----------
    metrics : List[str]
        List of benchmark metric names.
    """

    _channel: SimpleQueue

    def __init__(self, channel: SimpleQueue, metrics: List[str]):
        """
        Parameters
        ----------
        channel : SimpleQueue
            Channel to be used to send benchmarking events.
        metrics : List[str]
            List of benchmark metric names.
        """
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
    channel : SimpleQueue
        Channel on which it listens for benchmarking events.
    logger : DefaultLogger
        Logger for logging the benchmark process to the server log.
    name : str
        Name to identify a `BenchmarkListener` object.
    """

    benchmarker: bench.Benchmarker
    channel: SimpleQueue
    logger: DefaultLogger
    name: str
    _benchmarker: Benchmarker
    _thread: Thread

    def __init__(
        self,
        benchmarker: bench.Benchmarker,
        channel: SimpleQueue,
        logger: DefaultLogger,
        name: str = "benchmark_listener",
    ):
        """
        Parameters
        ----------
        benchmarker : src.benchmark.Benchmarker
            Benchmarker.
        channel : SimpleQueue
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
        self.logger.debug(f"[{self.name}] Started serving.")
        while True:
            entry = self.channel.get()
            if entry is None:
                break
            self.benchmarker.add(entry)

    def get_benchmarker(self) -> Benchmarker:
        """Gets a benchmarker to be sent to worker threads/processes

        Returns
        -------
        Benchmarker
            Benchmarker to be sent to worker threads/processes.
        """
        return self._benchmarker

    def terminate(self, _force: bool = False):
        """Stops serving the benchmark listener and terminates the benchmarking thread"""
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
