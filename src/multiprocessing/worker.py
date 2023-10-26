import logging
import torch.multiprocessing as tmp
from multiprocessing.managers import DictProxy
from queue import Queue
from threading import Condition
from typing import Any, Callable, Optional


class Worker:
    func: Callable
    module: Any
    channel: Queue
    result_pool: DictProxy
    logger: logging.Logger
    name: Optional[str]
    process: Optional[tmp.Process] = None

    def __init__(
        self,
        func: Callable,
        module: Any,
        channel: Queue,
        result_pool: DictProxy,
        logger: logging.Logger,
        name: Optional[str] = None,
    ):
        self.func = func
        self.module = module
        self.channel = channel
        self.result_pool = result_pool
        self.logger = logger
        self.name = name

    def start(self):
        self.process = tmp.Process(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self.process.start()

    def serve(self, module: Any):
        while True:
            self.logger.debug(f"[{self.name}] Waiting for a message.")
            key, cond, args = self.channel.get()
            self.result_pool[key] = self.func(*args, module=module)
            with cond:
                cond.notify()

    def terminate(self, force: bool = False):
        if self.process is None:
            raise ValueError("The worker process has not been started.")
        if not force:
            while not self.channel.empty():
                pass
        self.process.terminate()
        self.process.join()
