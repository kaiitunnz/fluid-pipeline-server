import os
import signal
import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

from src.logger import DefaultLogger


class Worker:
    """
    Pipeline worker.

    It serves the entire UI detection pipeline.

    Attributes
    ----------
    func : Callable
        Function to invoke the UI detection pipeline.
    channel : SimpleQueue
        Channel on which it listens for new jobs.
    name : Optional[str]
        Name of the instance, used to identify itself in the server log.
    module : Any
        UI detection pipeline, used by `func`.
    logger : Logger
        Logger to log its process.
    thread : Optional[threading.Thread]
        Associated worker thread.
    """

    func: Callable
    channel: SimpleQueue
    name: Optional[str]
    module: Any
    logger: DefaultLogger
    thread: Optional[threading.Thread]

    _server_pid: int

    def __init__(
        self,
        func: Callable,
        channel: SimpleQueue,
        module: Any,
        logger: DefaultLogger,
        server_pid: int,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        func : Callable
            Function to invoke the UI detection pipeline.
        channel : SimpleQueue
            Channel on which it listens for new jobs.
        module : Any
            UI detection pipeline, used by `func`.
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
        self.module = module
        self.logger = logger
        self._server_pid = server_pid
        self.thread = None

    def start(self):
        """Creates a pipeline worker thread and starts the worker"""
        self.thread = threading.Thread(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self.thread.start()

    def serve(self, module: Any):
        """Serves the pipeline

        Parameters
        ----------
        module : Any
            UI detection pipeline.
        """
        self.logger.debug(f"[{self.name}] Started serving.")
        try:
            while True:
                job = self.channel.get()
                if job is None:
                    raise EOFError
                out_channel, args = job
                assert isinstance(out_channel, SimpleQueue)
                out_channel.put(self.func(*args, module=module))
        except EOFError:
            self.logger.debug(f"[{self.name}] Channel closed.")
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error occured: {e}")
            os.kill(self._server_pid, signal.SIGTERM)

    def terminate(self, force: bool = False):
        """Terminates the pipeline worker

        Parameters
        ----------
        force : bool
            Whether to immediately terminate the worker thread without waiting for
            the pending jobs to finish.
        """
        if self.thread is None:
            return
        if not force:
            while not self.channel.empty():
                pass
        self.channel.put(None)
        self.thread.join()
        self.logger.debug(f"[{self.name}] Terminated.")
