from typing import Any, Sequence

from src.logger import ILogger
from src.multithread.worker import Worker


class LoadBalancer:
    """
    Load balancer.

    It balances the load on the workers of different instances of a UI detection
    module/component.

    Attributes
    ----------
    workers: Sequence[Worker]
        Sequence of workers of different instances of a single UI detection module/component.
    logger: ILogger
        Logger to log the UI detection process.
    """

    workers: Sequence[Worker]
    logger: ILogger

    def __init__(self, workers: Sequence[Worker], logger: ILogger):
        """
        Parameters
        ----------
        workers: Sequence[Worker]
            Sequence of workers of different instances of a single UI detection module/component.
        logger: ILogger
            Logger to log the UI detection process.
        """
        self.workers = workers
        self.logger = logger

    def send(self, job: Any):
        """Sends a job to a worker

        Parameters
        ----------
        job : Any
            Job to be processed by a worker.
        """
        worker = min(self.workers, key=lambda w: w.channel.qsize())
        worker.channel.put(job)

    def sendi(self, i: int, job: Any):
        """Sends a job to the worker specified by the index

        Parameters
        ----------
        i : int
            Index of the worker.
        job : Any
            Job to be processed by a worker.
        """
        self.workers[i].channel.put(job)

    def start(self):
        """Starts the workers managed by this load balancer"""
        for worker in self.workers:
            worker.start()

    def terminate(self, force: bool = False):
        """Terminates the pipeline workers"""
        for worker in self.workers:
            worker.terminate(force)
            self.logger.info(f"'{worker.name}' worker has terminated.")
