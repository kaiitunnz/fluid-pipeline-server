import logging
from multiprocessing.queues import SimpleQueue
from threading import Thread
from typing import Any, List

import src.benchmark as bench


class Benchmarker:
    _channel: SimpleQueue
    metrics: List[str]

    def __init__(self, channel: SimpleQueue, metrics: List[str]):
        self._channel = channel
        self.metrics = metrics

    def add(self, entry: List[Any]):
        self._channel.put(entry)


class BenchmarkListener:
    benchmarker: bench.Benchmarker
    channel: SimpleQueue
    logger: logging.Logger
    name: str
    _benchmarker: Benchmarker
    _thread: Thread

    def __init__(
        self,
        benchmarker: bench.Benchmarker,
        channel: SimpleQueue,
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
        self.logger.debug(f"[{self.name}] Started serving.")
        while True:
            entry = self.channel.get()
            if entry is None:
                break
            self.benchmarker.add(entry)

    def get_benchmarker(self):
        return self._benchmarker

    def terminate(self, _force: bool = False):
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
