import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

from src.hybrid.logging import Logger


class Worker:
    func: Callable
    channel: SimpleQueue
    name: Optional[str]
    _is_thread: bool
    module: Any
    logger: Logger
    _thread: Optional[threading.Thread] = None

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        module: Any,
        logger: Logger,
        name: Optional[str] = None,
        is_thread: bool = True,
    ):
        self.func = func
        self.channel = channel
        self.name = name
        self._is_thread = is_thread
        self.module = module
        self.logger = logger

    def start(self):
        if not self._is_thread:
            return
        self._thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self._thread.start()

    def serve(self, module: Any):
        self.logger.debug(f"[{self.name}] Started serving.")
        if not self._is_thread:
            return
        while True:
            msg = self.channel.get()
            if msg is None:
                return
            out_channel, args = msg
            assert isinstance(out_channel, SimpleQueue)
            out_channel.put(self.func(*args, module=module))

    def terminate(self, _force: bool = False):
        if not self._is_thread:
            self.logger.debug(f"[{self.name}] Terminated.")
            return
        if self._thread is None:
            raise ValueError("The worker thread has not been started.")
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
