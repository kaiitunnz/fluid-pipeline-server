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
    _thread: Optional[threading.Thread] = None

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
        self._thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self._thread.start()

    def serve(self, module: Any):
        try:
            while True:
                self.logger.debug(f"[{self.name}] Waiting for a message.")
                msg = self.channel.get()
                if msg is None:
                    return
                out_channel, args = msg
                assert isinstance(out_channel, SimpleQueue)
                out_channel.put(self.func(*args, module=module))
        except EOFError:
            self.logger.info(f"'{self.name}' worker's channel closed.")

    def terminate(self, _force: bool = False):
        if self._thread is None:
            raise ValueError("The worker thread has not been started.")
        self.channel.put(None)
        self._thread.join()
