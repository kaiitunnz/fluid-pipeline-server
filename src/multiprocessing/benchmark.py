import logging
from queue import Queue
from threading import Thread
from typing import Any, List

import src.benchmark as bench


class Benchmarker:
    _channel: Queue
    metrics: List[str]

    def __init__(self, channel: Queue, metrics: List[str]):
        self._channel = channel
        self.metrics = metrics

    def add(self, entry: List[Any]):
        self._channel.put(entry)


class BenchmarkListener:
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
        self.benchmarker = benchmarker
        self.channel = channel
        self.logger = logger
        self.name = name
        self._benchmarker = Benchmarker(channel, self.benchmarker.metrics)
        self._thread = Thread(target=self._serve, name="logger", daemon=False)

    def start(self):
        self._thread.start()

    def _serve(self):
        try:
            while True:
                entry = self.channel.get()
                if entry is None:
                    break
                self.benchmarker.add(entry)
        except EOFError:
            self.logger.info(f"'{self.name}' worker's channel closed.")

    def get_benchmarker(self):
        return self._benchmarker

    def terminate(self, _force: bool = False):
        try:
            self.channel.put(None)
        except BrokenPipeError:
            pass
        self._thread.join()
