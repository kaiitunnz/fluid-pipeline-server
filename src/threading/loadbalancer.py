from logging import Logger
from typing import Any, Sequence


from src.threading.worker import Worker


class LoadBalancer:
    workers: Sequence[Worker]
    logger: Logger

    def __init__(self, workers: Sequence[Worker], logger: Logger):
        self.workers = workers
        self.logger = logger

    def send(self, job: Any):
        worker = min(self.workers, key=lambda w: w.channel.qsize())
        worker.channel.put(job)

    def sendi(self, i: int, job: Any):
        self.workers[i].channel.put(job)

    def start(self):
        for worker in self.workers:
            worker.start()

    def terminate(self, force: bool = False):
        for worker in self.workers:
            worker.terminate(force)
            self.logger.info(f"'{worker.name}' worker has terminated.")
