import logging
import multiprocessing as mp
import os
import signal
import socket as sock
import sys
import time
from threading import Semaphore
from typing import List, Optional

from PIL import ImageFile

from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.hybrid.benchmark import BenchmarkListener
from src.hybrid.handler import ConnectionHandler
from src.hybrid.logging import LogListener
from src.pipeline import PipelineServerInterface

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


class PipelineServer(PipelineServerInterface):
    hostname: str
    port: str
    pipeline: PipelineConstructor
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    verbose: bool
    logger: logging.Logger
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
    ):
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.num_instances = num_instances
        self.verbose = verbose
        self.logger = self._init_logger(verbose)
        self.handlers = []

        benchmark_metrics = [
            "Waiting time",
            "UI detection time",
            "Invalid UI detection time",
            "UI matching time",
            "UI processing time",
            "Text recognition time",
            "Icon labeling time",
            "Processing time",
        ]
        self.benchmarker = (
            Benchmarker(
                benchmark_metrics,
                benchmark_file or DEFAULT_BENCHMARK_FILE,
            )
            if benchmark
            else None
        )

    def start(self, warmup_image: Optional[str] = None):
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

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        self.logger.info(
            f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
        )

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
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)

    @classmethod
    def _init_logger(cls, verbose: bool = True) -> logging.Logger:
        fmt = "[%(asctime)s | %(name)s] [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(format=fmt, datefmt=datefmt)
        logger = logging.getLogger(cls.__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger
