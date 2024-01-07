import multiprocessing as mp
import os
import signal
import socket as sock
import sys
import threading
from multiprocessing.queues import SimpleQueue
from multiprocessing.synchronize import Semaphore
from queue import Queue
from threading import Thread
from typing import Any, List, Optional, Tuple

from src.constructor import PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.helper import PipelineHelper
from src.hybrid.logger import Logger
from src.hybrid.manager import PipelineManager

Job = Tuple[int, float, sock.socket]


class _HandlerHelper:
    """
    Helper that manages connection handling threads.

    Attributes
    ----------
    helper : PipelineHelper
        Helper used to access the UI detection pipeline modules.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    job_queue : Queue
        Queue on which the connection handling threads listening.
    logger : Logger
        Logger for logging the UI detection process.
    name : str
        Name of the connection handler process.
    test_mode : bool
        Whether to handle connections in test mode.
    """

    helper: PipelineHelper
    chunk_size: int
    max_image_size: int
    job_queue: Queue
    logger: Logger
    name: str
    test_mode: bool

    _workers: List[Thread]

    def __init__(
        self,
        helper: PipelineHelper,
        chunk_size: int,
        max_image_size: int,
        num_workers: int,
        logger: Logger,
        name: str,
        test_mode: bool,
    ):
        """
        Parameters
        ----------
        helper : PipelineHelper
            Helper used to access the UI detection pipeline modules.
        chunk_size : int
            Chunk size for reading bytes from the sockets.
        max_image_size : int
            Maximum size of an image from the client.
        num_workers : int
            Number of connection handling threads to be spawned.
        logger : Logger
            Logger for logging the UI detection process.
        name : str
            Name of the connection handler process.
        test_mode : bool
            Whether to handle connections in test mode.
        """
        self.helper = helper
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.job_queue = Queue(num_workers)
        self.logger = logger
        self.name = name
        self.test_mode = test_mode

        self._workers = [
            Thread(target=self._serve, args=(self.job_queue,), daemon=False)
            for _ in range(num_workers)
        ]

    def start(self):
        """Starts the conenction handling threads"""
        for worker in self._workers:
            worker.start()

    def _serve(self, job_queue: Queue):
        """Listens to the job queue and handles incoming jobs

        Parameters
        ----------
        job_queue : Queue
            Queue to be listened on for incoming jobs.
        """
        while True:
            job = job_queue.get()
            if job is None:
                break
            self._handle_connection(*job)

    def _handle_connection(
        self,
        job_no: int,
        start_time: float,
        conn: sock.socket,
    ):
        """Handles a job/connection, essentially serving the UI detection pipeline

        Parameters
        ----------
        job_no : int
            Job number, used to identify the job.
        start_time : float
            Time at which the connection is accepted.
        conn : socket
            Socket for an accepted connection.
        """
        self.helper.serve(
            job_no,
            start_time,
            conn,
            self.chunk_size,
            self.max_image_size,
            self.test_mode,
        )

    def terminate(self, _force: bool = False):
        """Terminates the connection handling threads

        It blocks until all the threads finish handling the current jobs.
        """
        for _ in range(len(self._workers)):
            self.job_queue.put(None)
        for i, worker in enumerate(self._workers):
            worker.join()
            self.logger.debug(f"[{self.name}] Worker{i} terminated.")


class ConnectionHandler:
    """
    A class that handles incoming requests and serves a single instance of the UI
    detection pipeline.

    It handles multiple requests concurrently by creating a pool of worker threads.
    However, it only manages one instance of the UI detection pipeline, whose usage
    is shared by the worker threads.

    Attributes
    ----------
    key : int
        ID of the instance. It is used to differentiate the instance from other instances
        of this class.
    name : str
        Name of the instance. There can be other instances of the same name.
    constructor : PipelineConstructor
        Constructor of the UI detection pipeline to be served.
    num_workers : int
        Number of worker threads in the pool to handle incoming requests.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the UI detection pipeline server.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    """

    CLS: str = "connection_handler"

    key: int
    name: str
    constructor: PipelineConstructor
    num_workers: int
    logger: Logger
    benchmarker: Optional[Benchmarker]

    chunk_size: int
    max_image_size: int

    _server_pid: int
    _job_queue: SimpleQueue[Optional[Job]]
    _process: Optional[mp.Process]
    _ready_sema: Semaphore
    _is_ready: Any
    _exit_sema: threading.Semaphore  # To handle terminating signals only once

    def __init__(
        self,
        key: int,
        constructor: PipelineConstructor,
        job_queue: SimpleQueue[Optional[Job]],
        num_workers: int,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        server_pid: int,
        chunk_size: int,
        max_image_size: int,
        name: str = CLS,
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other instances
            of this class.
        constructor : PipelineConstructor
            Constructor of the UI detection pipeline to be served.
        job_queue : SimpleQueue[Optional[Job]]
            Queue on which to listen for new jobs.
        num_workers : int
            Number of worker threads in the pool to handle incoming requests.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker for benchmarking the UI detection pipeline server.
        server_pid : int
            Process ID of the pipeline server.
        chunk_size : int
            Chunk size for reading bytes from the sockets.
        max_image_size : int
            Maximum size of an image from the client.
        name : str
            Name of the instance.
        """
        self.key = key
        self.constructor = constructor
        self.num_workers = num_workers
        self.logger = logger
        self.benchmarker = benchmarker
        self.name = name
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size

        self._server_pid = server_pid
        self._job_queue = job_queue
        self._process = None
        self._ready_sema = mp.Semaphore(0)
        self._is_ready = mp.Value("i", 0, lock=False)
        self._exit_sema: threading.Semaphore = threading.Semaphore(1)

    def start(self, warmup_image: Optional[str] = None):
        """Starts the connection handler process

        Parameters
        ----------
        warmup_image : Optional[str]
            Path to the image for warming up the UI detection pipeline and performing
            initial testing. `None` to skip the warming up stage.
        """
        self._process = mp.Process(
            target=self._serve,
            name=self.name,
            args=(warmup_image,),
            daemon=False,
        )
        self._process.start()

    def _serve(self, warmup_image: Optional[str] = None):
        """Initializes the UI detection pipeline and serves incoming requests

        Parameters
        ----------
        warmup_image : Optional[str]
            Path to the image for warming up the UI detection pipeline and performing
            initial testing. `None` to skip the warming up stage.
        """
        self.logger.debug(f"[{self.get_name()}] Started serving.")

        manager = PipelineManager(
            self.key,
            self.constructor,
            self.logger,
            self.benchmarker,
            self._server_pid,
        )
        handler_helper = _HandlerHelper(
            manager.get_helper(),
            self.chunk_size,
            self.max_image_size,
            self.num_workers,
            self.logger,
            self.get_name(),
            self.constructor.test_mode,
        )

        manager.start()
        handler_helper.start()

        self._register_signal_handlers(handler_helper, manager)
        if warmup_image is not None:
            self._warmup(manager.get_helper(), warmup_image)
        self._ready()

        while True:
            job = self._job_queue.get()
            if job is None:
                break
            handler_helper.job_queue.put(job, block=True)

    def _ready(self):
        """Notifies waiters that this connection handler is ready to serve"""
        self._is_ready.value = 1
        self._ready_sema.release()

    def wait_ready(self) -> bool:
        """Waits for this connection handler to be ready to serve

        Returns
        -------
        bool
            Whether this connection handler is ready to serve. `False` if it failed
            to start.
        """
        self._ready_sema.acquire()
        return bool(self._is_ready.value)

    @staticmethod
    def send(
        job_queue: SimpleQueue,
        job_no: int,
        start_time: float,
        conn: sock.socket,
    ):
        """Sends a job to connection handlers listening on a job queue

        Parameters
        ----------
        job_queue : SimpleQueue
            Queue on which connection handlers are listening.
        job_no : int
            Job number, used to identify the job.
        start_time: float
            Time at which the connection is accepted.
        conn : socket
            Socket for an accepted connection.
        """
        job_queue.put((job_no, start_time, conn))

    def _error(self, message: str, info: Any):
        """Logs an error event and marks this connection handler as not ready to serve

        Parameters
        ----------
        message : str
            Error message.
        info : Any
            Additional information about the error.
        """
        self.logger.error(message)
        self.logger.debug("Cause:")
        self.logger.debug(str(info))
        self._is_ready.value = 0

    def _warmup(self, helper: PipelineHelper, warmup_image: str, kill: bool = True):
        """Warms up the UI detection pipeline and performs initial testing

        Parameters
        ----------
        helper : PipelineHelper
            Helper used to access the UI detection pipeline modules.
        warmup_image : str
            Path to the image for warming up the UI detection pipeline and performing
            initial testing.
        kill : bool
            Whether to kill the process if an error occurs.
        """
        success = helper.warmup(warmup_image, self.get_name(), self._error)

        if success:
            self.logger.debug(f"[{self.get_name()}] Warm-up complete.")
        else:
            self.logger.debug(f"[{self.get_name()}] Sending termination signal.")
            self._ready_sema.release()
            if kill:
                os.kill(os.getpid(), signal.SIGTERM)

    def _register_signal_handlers(
        self, helper: _HandlerHelper, manager: PipelineManager
    ):
        """Registers signal handlers to handle termination signals

        Parameters
        ----------
        helper : _HandlerHelper
            Helper used to access the connection handling threads.
        manager : PipelineManager
            Manager used to access the worker threads serving UI detection pipeline
            modules.
        """
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.debug(
                f"[{self.get_name()}] Termination signal received: {signal.Signals(signum).name}"
            )
            helper.terminate(True)
            manager.terminate(True)
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)

    def get_name(self) -> str:
        """Gets the name of this instance

        Returns
        -------
        str
            Name of this instance.
        """
        return self.name + str(self.key)

    def terminate(self, force: bool = False):
        """Terminates the connection handler process

        Parameters
        ----------
        force : bool
            Whether to immediately stop the process. If `False`, waits until all the
            preceding jobs are handled.
        """
        if self._process is None:
            return
        if force:
            self._process.terminate()
            self._process.join(2)
            if self._process.exitcode is None:
                self.logger.debug(f"[{self.get_name()}] Joining timeout.")
            self._process.kill()
        else:
            self._job_queue.put(None)
        self._process.join()
        self.logger.debug(f"[{self.get_name()}] Terminated.")
