import os
import signal
import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

from src.constructor import ModuleConstructor
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
    constructor: ModuleConstructor
    logger: Logger
    _server_pid: int

    module: Optional[Any]
    _thread: Optional[threading.Thread]

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        constructor: ModuleConstructor,
        logger: Logger,
        server_pid: int,
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
        contructor : ModuleConstructor
            Constructor of a UI detection pipeline component/module.
        logger : Logger
            Logger to log its process.
        server_pid : int
            Process ID of the pipeline server.
        name : Optional[str]
            Name of the instance, used to identify itself in the server log.
        is_thread : bool
            Whether to spawn a thread to serve the pipeline component.
        """
        self.func = func
        self.channel = channel
        self.name = name
        self._is_thread = is_thread
        self.constructor = constructor
        self.logger = logger
        self._server_pid = server_pid
        self.module = None
        self._thread = None

    def start(self):
        """Starts the pipeline worker

        It creates a worker thread if `is_thread` is true. Otherwise, it serves the
        component as a function call on the caller thread.
        """
        if not self._is_thread:
            self.module = self.constructor()
            return
        self._thread = threading.Thread(target=self.serve, name=self.name, daemon=False)
        self._thread.start()

    def serve(self):
        """Serves the pipeline component if `is_thread` is true

        Parameters
        ----------
        module : Any
            Pipeline module/component to be served.
        """
        self.logger.debug(f"[{self.name}] Started serving.")
        try:
            self.module = self.constructor()
            if not self._is_thread:
                return
            while True:
                msg = self.channel.get()
                if msg is None:
                    return
                out_channel, args = msg
                assert isinstance(out_channel, SimpleQueue)
                out_channel.put(self.func(*args, module=self.module))
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error occured: {e}")
            os.kill(self._server_pid, signal.SIGTERM)

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
