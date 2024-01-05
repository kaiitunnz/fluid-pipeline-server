import os
import signal
import socket as sock
import sys
import time
from multiprocessing.pool import Pool, ThreadPool
from typing import Any, Optional

from PIL import ImageFile  # type: ignore

from src.benchmark import BENCHMARK_METRICS, Benchmarker
from src.constructor import PipelineConstructor
from src.multithread.helper import PipelineHelper
from src.multithread.logger import Logger
from src.multithread.manager import PipelineManager
from src.pipeline import IPipelineServer
from src.process import ui_detection_serve, ui_detection_warmup

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


class PipelineServer(IPipelineServer):
    """
    UI detection pipeline server with multithreading.

    It utilizes multithreading to handle client connections concurrently. It initializes
    exactly one instance of the UI detection pipeline, each of whose components are
    executing in separate threads. Threads communicate via channels to send data
    to be processed and receive the results.

    Attributes
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
        Number of connection handling threads.
    socket : Optional[sock.socket]
        Server socket.
    verbose : bool
        Whether to log server events verbosely.
    logger : logging.Logger
        Logger to log server events.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the server.
    """

    hostname: str
    port: str
    pipeline: PipelineConstructor
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    verbose: bool
    logger: Logger
    benchmarker: Optional[Benchmarker]

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
            Number of connection handling threads.
        num_instances : int
            Number of instances of the UI detection pipeline.
        verbose : bool
            Whether to log server events verbosely.
        benchmark : bool
            Whether to run the server in the benchmark mode.
        benchmark_file : Optional[str]
            Path to the file to save the benchmark results.
        """
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.num_instances = num_instances
        self.verbose = verbose
        self.logger = Logger(PipelineServer._init_logger(verbose))

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

        manager = PipelineManager(
            self.pipeline, self.logger, self.benchmarker, self.num_instances
        )
        manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool, manager)
            if warmup_image is not None:
                self._warmup(manager, warmup_image)

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
                pool.apply_async(
                    ui_detection_serve,
                    args=(
                        manager.get_helper(),
                        job_no,
                        time.time(),
                        conn,
                        self.chunk_size,
                        self.max_image_size,
                        self.pipeline.test_mode,
                    ),
                )

    def _warmup(self, manager: PipelineManager, warmup_image: str, kill: bool = True):
        """Warms up the UI detection pipeline and performs initial testing

        Parameters
        ----------
        manager : PipelineManager
            Manager of the UI detection pipeline.
        warmup_image : str
            Path to the image for warming up the UI detection pipeline and performing
            initial testing.
        kill : bool
            Whether to kill the process if an error occurs.
        """

        def on_error(message: str, info: Any):
            self.logger.error(message)
            self.logger.debug("Cause:")
            self.logger.debug(str(info))

        success = True
        for i in range(self.num_instances):
            name = f"pipeline{i}"
            success &= ui_detection_warmup(
                manager.get_warmup_helper(i), warmup_image, name, on_error
            )
            if not success:
                break

        if success:
            self.logger.debug("Warm-up complete.")
        else:
            self.logger.debug("Sending termination signal.")
            if kill:
                os.kill(os.getpid(), signal.SIGTERM)

    def _register_signal_handlers(self, pool: Pool, manager: PipelineManager):
        """Registers signal handlers to handle termination signals

        Parameters
        ----------
        pool : Pool
            Pool of connection handling processes.
        manager : PipelineManager
            Manager of the resources used by the UI detection pipeline.
        """
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            manager.terminate(True)
            self.logger.info("Server successfully exited.")
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)
