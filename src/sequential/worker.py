import logging
import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional


class Worker:
    func: Callable
    channel: SimpleQueue
    name: Optional[str]
    module: Any
    logger: logging.Logger
    thread: Optional[threading.Thread] = None

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        module: Any,
        logger: logging.Logger,
        name: Optional[str] = None,
    ):
        self.func = func
        self.channel = channel
        self.name = name
        self.module = module
        self.logger = logger

    def start(self):
        self.thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self.thread.start()

    def serve(self, module: Any):
        while True:
            self.logger.debug(f"[{self.name}] Waiting for a message.")
            out_channel, args = self.channel.get()
            assert isinstance(out_channel, SimpleQueue)
            out_channel.put(self.func(*args, module=module))

    def terminate(self, force: bool = False):
        if self.thread is None:
            raise ValueError("The worker thread has not been started.")
        if not force:
            while not self.channel.empty():
                pass
        self.thread.join()
