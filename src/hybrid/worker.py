import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

from src.hybrid.logger import Logger


class Worker:
    """
    Pipeline worker.

    It serves a component of the UI detection pipeline.

    Attributes
    ----------
    func : Callable
        Function to invoke the pipeline component.
    channel : SimpleQueue
        Channel on which it listens for new jobs.
    name : Optional[str]
        Name of the instance, used to identify itself in the server log.
    module : Any
        UI detection pipeline component/module, used by `func`.
    logger : Logger
        Logger to log its process.
    """

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
        """
        Parameters
        ----------
        func : Callable
            Function to invoke the pipeline component.
        channel : SimpleQueue
            Channel on which it listens for new jobs.
        module : Any
            UI detection pipeline component/module, used by `func`.
        logger : Logger
            Logger to log its process.
        name : Optional[str]
            Name of the instance, used to identify itself in the server log.
        is_thread : bool
            Whether to spawn a thread to serve the pipeline component.
        """
        self.func = func
        self.channel = channel
        self.name = name
        self._is_thread = is_thread
        self.module = module
        self.logger = logger

    def start(self):
        """Starts the pipeline worker

        It creates a worker thread if `is_thread` is true. Otherwise, it serves the
        component as a function call on the caller thread.
        """
        if not self._is_thread:
            return
        self._thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self._thread.start()

    def serve(self, module: Any):
        """Serves the pipeline component if `is_thread` is true

        Parameters
        ----------
        module : Any
            Pipeline module/component to be served.
        """
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
        """Terminates the pipeline worker.

        It waits until all the pending work finishes.
        """
        if not self._is_thread:
            self.logger.debug(f"[{self.name}] Terminated.")
            return
        if self._thread is None:
            raise ValueError("The worker thread has not been started.")
        self.channel.put(None)
        self._thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
