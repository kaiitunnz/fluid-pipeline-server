import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

from src.logger import ILogger


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
    module: Any
    logger: ILogger
    _thread: Optional[threading.Thread] = None

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        module: Any,
        logger: ILogger,
        name: Optional[str] = None,
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
        """
        self.func = func
        self.channel = channel
        self.name = name
        self.module = module
        self.logger = logger

    def start(self):
        """Creates a pipeline worker thread and starts the worker"""
        self._thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self._thread.start()

    def serve(self, module: Any):
        """Serves the pipeline component

        Parameters
        ----------
        module : Any
            Pipeline module/component to be served.
        """
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
        """Terminates the pipeline worker

        It waits until all the pending work finishes.
        """
        if self._thread is None:
            raise ValueError("The worker thread has not been started.")
        self.channel.put(None)
        self._thread.join()
