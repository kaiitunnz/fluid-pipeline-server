import logging
import multiprocessing as mp
import os
import signal
import socket as sock
import sys
from multiprocessing.managers import SyncManager
from typing import List, Optional

from PIL import ImageFile

from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.hybrid.benchmark import BenchmarkListener
from src.hybrid.handler import ConnectionHandler
from src.hybrid.logging import LogListener

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"
SAVE_IMG_DIR = "res"
SAVE_IMG = False


class PipelineServer:
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
    _is_exit: bool = False

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
        self._is_exit = False

        manager = mp.Manager()
        job_queue = manager.Queue()
        log_listener = LogListener(self.logger, manager.Queue())
        benchmark_listener = (
            None
            if self.benchmarker is None
            else BenchmarkListener(self.benchmarker, manager.Queue(), self.logger)
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

        self._register_signal_handlers(manager, log_listener, benchmark_listener)

        log_listener.start()
        if benchmark_listener is not None:
            benchmark_listener.start()
        for handler in self.handlers:
            handler.start(warmup_image)

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        self.logger.info(
            f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
        )
        while True:
            conn, addr = self.socket.accept()
            self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
            job_queue.put(conn)

    def _register_signal_handlers(
        self,
        manager: SyncManager,
        log_listener: LogListener,
        benchmark_listener: Optional[BenchmarkListener],
    ):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if self._is_exit:
                return
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            for handler in self.handlers:
                handler.terminate(force=True)
                self.logger.info(f"[{handler.get_name()}] Terminated.")
            log_listener.terminate(True)
            self.logger.info(f"[{log_listener.name}] Terminated.")
            if benchmark_listener is not None:
                benchmark_listener.terminate(True)
                self.logger.info(f"[{benchmark_listener.name}] Terminated.")
            manager.shutdown()
            manager.join()
            self.logger.info("Server successfully exited.")
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
