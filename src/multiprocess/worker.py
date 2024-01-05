import os
import torch.multiprocessing as tmp
from multiprocessing.managers import DictProxy
from queue import Queue
from typing import Any, Callable, Optional

from src.multiprocess.logger import Logger
from src.pipeline import PipelineModule


class Worker:
    """
    Pipeline worker.

    It serves a component of the UI detection pipeline.

    Attributes
    ----------
    func : Callable
        Function to invoke the pipeline component.
    constructor : Callable[[], Any]
        Constructor of the pipeline component.
    channel : Queue
        Channel on which it listens for new jobs.
    result_pool : DictProxy
        Pool to which it posts its outputs. Each output is associated with a unique
        key, which can be used by the consumer to retrieve a specific result.
    logger : Logger
        Logger to log its process.
    name: Optional[str]
        Name of the instance, used to identify itself in the server log.
    process : Optional[tmp.Process]
        Associated worker process.
    """

    func: Callable
    constructor: Callable[[], Any]
    channel: Queue
    result_pool: DictProxy
    logger: Logger
    name: Optional[str]
    process: Optional[tmp.Process] = None

    def __init__(
        self,
        func: Callable,
        constructor: Callable[[], Any],
        channel: Queue,
        result_pool: DictProxy,
        logger: Logger,
        name: Optional[PipelineModule] = None,
    ):
        """
        Parameters
        ----------
        func : Callable
            Function to invoke the pipeline component.
        constructor : Callable[[], Any]
            Constructor of the pipeline component.
        channel : Queue
            Channel on which it listens for new jobs.
        result_pool : DictProxy
            Pool to which it posts its outputs. Each output is associated with a unique
            key, which can be used by the consumer to retrieve a specific result.
        logger : Logger
            Logger to log its process.
        name: Optional[str]
            Name of the instance, used to identify itself in the server log.
        """
        self.func = func
        self.constructor = constructor
        self.channel = channel
        self.result_pool = result_pool
        self.logger = logger
        self.name = None if name is None else name.value

    def start(self):
        """Starts the pipeline worker"""
        self.process = tmp.Process(
            target=self.serve,
            name=self.name,
            args=(self.constructor,),
            daemon=False,
        )
        self.process.start()

    def serve(self, constructor: Callable[[], Any]):
        """Serves the pipeline component

        Parameters
        ----------
        constructor : Callable[[], Any]
            Constructor of the pipeline component.
        """
        self.logger.debug(f"[{self.name}] Start serving (PID={os.getpid()}).")
        module = constructor()
        try:
            while True:
                self.logger.debug(f"[{self.name}] Waiting for a message.")
                key, cond, args = self.channel.get()
                self.result_pool[key] = self.func(*args, module=module)
                with cond:
                    cond.notify()
        except EOFError:
            self.logger.info(f"'{self.name}' worker's channel closed.")

    def terminate(self, force: bool = False):
        """Terminates the pipeline worker

        Parameters
        ----------
        force : bool
            Whether to immediately terminate the worker process without waiting for
            the pending jobs to finish.
        """
        if self.process is None:
            raise ValueError("The worker process has not been started.")
        if not force:
            while not self.channel.empty():
                pass
        self.process.terminate()
        self.process.join()