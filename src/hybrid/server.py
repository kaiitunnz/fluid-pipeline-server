import multiprocessing as mp
import os
import signal
import time
from PIL import ImageFile  # typing: ignore
from threading import Semaphore
from typing import List, Optional

from src.benchmark import BENCHMARK_METRICS, Benchmarker
from src.constructor import PipelineConstructor
from src.logger import DefaultLogger
from src.hybrid.benchmark import BenchmarkListener
from src.hybrid.handler import ConnectionHandler
from src.hybrid.logger import LogListener
from src.pipeline import IPipelineServer

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


class PipelineServer(IPipelineServer):
    """
    UI detection pipeline server with a hybrid architecture.

    It utilizes both multithreading, to minimize communication overhead, and multiprocessing,
    to fully leverage parallel processors. It comprises a main process, which accepts
    connections from clients and distributes them to connection handler processes,
    each having worker threads to concurrently handle requests distributed to it.
    Each connection handler process has exactly one instance of the UI detection pipeline,
    whose usage is shared by the worker threads within the worker process.

    Attributes
    ----------
    hostname : str
        Host name.
    port : str
        Port to listen to client connections.
    socket : Optional[sock.socket]
        Server socket.
    pipeline: PipelineConstructor
        Constructor of the UI detection pipeline.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    num_workers : int
        Number of worker threads in each connection handler process.
    verbose : bool
        Whether to log server events verbosely.
    logger : Logger
        Logger to log server events.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the server.
    handlers : List[ConnectionHandler]
        Connection handlers, each representing a connection handler process.
    """

    pipeline: PipelineConstructor
    chunk_size: int
    max_image_size: int
    num_workers: int
    verbose: bool
    logger: DefaultLogger
    benchmarker: Optional[Benchmarker]

    handlers: List[ConnectionHandler]
    _exit_sema: Semaphore = Semaphore(1)  # To handle terminating signals only once

    def __init__(
        self,
        *,
        hostname: str,
        port: str,
        pipeline: PipelineConstructor,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
        num_instances: int = 1,
        verbose: bool = True,
        benchmark: bool = False,
        benchmark_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        hostname : str
            Host name.
        port : str
            Port to listen to client connections.
        pipeline: PipelineConstructor
            Constructor of the UI detection pipeline.
        chunk_size : int
            Chunk size for reading bytes from the sockets.
        max_image_size : int
            Maximum size of an image from the client.
        num_workers : int
            Number of worker threads in each connection handler process.
        num_instances : int
            Number of connection handler processes, which is equivalent to the number
            of instances of the UI detection pipeline.
        verbose : bool
            Whether to log server events verbosely.
        benchmark : bool
            Whether to run the server in the benchmark mode.
        benchmark_file : Optional[str]
            Path to the file to save the benchmark results.
        """
        super().__init__(
            hostname, port, None, DefaultLogger(PipelineServer._init_logger(verbose))
        )
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.num_instances = num_instances
        self.verbose = verbose
        self.handlers = []

        self.benchmarker = (
            Benchmarker(
                BENCHMARK_METRICS,
                benchmark_file or DEFAULT_BENCHMARK_FILE,
            )
            if benchmark
            else None
        )

        if len(kwargs) > 0:
            self.logger.warn(f"Got unexpected arguments: {kwargs}")

    def start(self, warmup_image: Optional[str] = None):
        """Starts the UI detection pipeline server

        Parameters
        ----------
        warmup_image: Optional[str]
            Path to the image for warming up the UI detection pipeline and performing
            initial testing. `None` to skip the warming up stage.
        """
        self.logger.info("Starting the pipeline server...")

        # Use SimpleQueue instead of Queue, which has its own finalizer
        # that closes itself before all log events are acknowledged.
        job_queue = mp.SimpleQueue()
        log_listener = LogListener(self.logger, mp.SimpleQueue())
        benchmark_listener = (
            None
            if self.benchmarker is None
            else BenchmarkListener(self.benchmarker, mp.SimpleQueue(), self.logger)
        )

        for i in range(self.num_instances):
            self.handlers.append(
                ConnectionHandler(
                    i,
                    self.pipeline,
                    job_queue,
                    self.num_workers,
                    log_listener.get_logger(),
                    None
                    if benchmark_listener is None
                    else benchmark_listener.get_benchmarker(),
                    self.chunk_size,
                    self.max_image_size,
                )
            )

        self._register_signal_handlers(log_listener, benchmark_listener)

        log_listener.start()
        if benchmark_listener is not None:
            benchmark_listener.start()
        for handler in self.handlers:
            handler.start(warmup_image)
        for handler in self.handlers:
            if not handler.wait_ready():
                self.logger.error("Failed to start the pipeline server.")
                os.kill(os.getpid(), signal.SIGTERM)

        self.socket = self.bind()

        job_no = 0
        while True:
            conn, addr = self.socket.accept()
            self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
            job_no += 1
            ConnectionHandler.send(job_queue, job_no, time.time(), conn)

    def _register_signal_handlers(
        self,
        log_listener: LogListener,
        benchmark_listener: Optional[BenchmarkListener],
    ):
        """Registers signal handlers to handle termination signals

        Parameters
        ----------
        log_listener : LogListener
            Listener of logging events.
        benchmark_listener : Optional[BenchmarkListener],
            Listener of benchmarking events.
        """
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            self.logger.info("Terminating the pipeline server...")
            for handler in self.handlers:
                handler.terminate(force=True)
            log_listener.terminate(True)
            if benchmark_listener is not None:
                benchmark_listener.terminate(True)
            self.logger.info("Server exited successfully.")
            self.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)
