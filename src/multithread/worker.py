import os
import signal
import threading
from queue import SimpleQueue
from typing import Callable, Optional

from src.constructor import ModuleConstructor
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
    constructor: ModuleConstructor
    logger: ILogger
    _server_pid: int
    _thread: Optional[threading.Thread]

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        constructor: ModuleConstructor,
        logger: ILogger,
        server_pid: int,
        name: Optional[str] = None,
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
        """
        self.func = func
        self.channel = channel
        self.name = name
        self.constructor = constructor
        self.logger = logger
        self._server_pid = server_pid
        self._thread = None

    def start(self):
        """Creates a pipeline worker thread and starts the worker"""
        self._thread = threading.Thread(target=self.serve, name=self.name, daemon=False)
        self._thread.start()

    def serve(self):
        """Serves the pipeline component

        Parameters
        ----------
        module : Any
            Pipeline module/component to be served.
        """
        try:
            module = self.constructor()
            try:
                while True:
                    msg = self.channel.get()
                    if msg is None:
                        return
                    out_channel, args = msg
                    assert isinstance(out_channel, SimpleQueue)
                    out_channel.put(self.func(*args, module=module))
            except EOFError:
                self.logger.info(f"'{self.name}' worker's channel closed.")
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error occured: {e}")
            os.kill(self._server_pid, signal.SIGTERM)

    def terminate(self, _force: bool = False):
        """Terminates the pipeline worker

        It waits until all the pending work finishes.
        """
        if self._thread is None:
            raise ValueError("The worker thread has not been started.")
        self.channel.put(None)
        self._thread.join()
