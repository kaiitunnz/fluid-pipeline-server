import os
import torch.multiprocessing as tmp
from multiprocessing.managers import DictProxy
from queue import Queue
from typing import Any, Callable, Optional

from src.multiprocessing.logging import Logger


class Worker:
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
        name: Optional[str] = None,
    ):
        self.func = func
        self.constructor = constructor
        self.channel = channel
        self.result_pool = result_pool
        self.logger = logger
        self.name = name

    def start(self):
        self.process = tmp.Process(
            target=self.serve, name=self.name, args=(self.constructor,), daemon=False
        )
        self.process.start()

    def serve(self, constructor: Callable[[], Any]):
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
        if self.process is None:
            raise ValueError("The worker process has not been started.")
        if not force:
            while not self.channel.empty():
                pass
        self.process.terminate()
        self.process.join()
