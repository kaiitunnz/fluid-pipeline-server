import os
import signal
import time
from threading import Semaphore
from typing import Any, Optional

import torch
import torch.multiprocessing as tmp
from PIL import ImageFile  # type: ignore
from multiprocessing.pool import Pool

from src.benchmark import BENCHMARK_METRICS, Benchmarker
from src.constructor import PipelineConstructor
from src.logger import DefaultLogger
from src.multiprocess.manager import PipelineHelper, PipelineManager
from src.pipeline import IPipelineServer

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"
SAVE_IMG_DIR = "res"
SAVE_IMG = False


class PipelineServer(IPipelineServer):
    """
    UI detection pipeline server with multiprocessing.

    It utilizes multiprocessing to handle client connections concurrently. It initializes
    exactly one instance of the UI detection pipeline, each of whose components are
    executing in separate processes. Data to be processed are sent to the components
    via channels, and the results are retrieved from the result pools.

    Attributes
    ----------
    hostname : str
        Host name.
    port : int
        Port to listen to client connections.
    socket : Optional[sock.socket] = None
        Server socket.
    pipeline : PipelineConstructor
        Constructor of the UI detection pipeline.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    num_workers : int
        Number of connection handling processes.
    verbose : bool
        Whether to log server events verbosely.
    logger : DefaultLogger
        Logger to log server events.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the server.
    """

    pipeline: PipelineConstructor
    chunk_size: int
    max_image_size: int
    num_workers: int
    verbose: bool
    logger: DefaultLogger
    benchmarker: Optional[Benchmarker]
    _exit_sema: Semaphore  # To handle terminating signals only once

    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        pipeline: PipelineConstructor,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
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
        port : int
            Port to listen to client connections.
        pipeline : PipelineConstructor
            Constructor of the UI detection pipeline.
        chunk_size : int
            Chunk size for reading bytes from the sockets.
        max_image_size : int
            Maximum size of an image from the client.
        num_workers : int
            Number of connection handling processes.
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
        self.verbose = verbose
        self._exit_sema = Semaphore(1)

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

        # Required to prevent PyTorch from hanging
        # https://github.com/pytorch/pytorch/issues/82843
        torch.set_num_threads(1)

        if torch.cuda.is_available():
            tmp.set_start_method("forkserver", force=True)  # To enable CUDA.

        with tmp.Manager() as sync_manager:
            manager = PipelineManager(
                self.pipeline,
                sync_manager,
                self.logger,
                self.benchmarker,
                self.getpid(),
            )
            manager.start()

            with tmp.Pool(processes=self.num_workers) as pool:
                self._register_signal_handlers(pool, manager)

                if warmup_image is not None:
                    self._warmup(manager.get_helper(), warmup_image)

                self.socket = self.bind()
                if self.socket is None:
                    return

                job_no = 0
                while True:
                    conn, addr = self.socket.accept()
                    self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
                    job_no += 1
                    pool.apply_async(
                        manager.get_helper().serve,
                        args=(
                            job_no,
                            time.time(),
                            conn,
                            self.chunk_size,
                            self.max_image_size,
                            self.pipeline.test_mode,
                        ),
                    )

    def _warmup(self, helper: PipelineHelper, warmup_image: str, kill: bool = True):
        """Warms up the UI detection pipeline and performs initial testing

        Parameters
        ----------
        helper : PipelineHelper
            Helper for accessing the UI detection pipeline modules.
        warmup_image : str
            Path to the image for warming up the UI detection pipeline and performing
            initial testing.
        kill : bool
            Whether to kill the process if an error occurs.
        """

        def on_error(message: str, info: Any):
            self.logger.error(f"{message} | Cause: {str(info)}")

        name = "Pipeline"
        try:
            success = helper.warmup(warmup_image, name, on_error)
        except Exception as e:
            success = False
            on_error("Exception occured when warming up", e)

        if success:
            self.logger.debug(f"[{name}] Warm-up complete.")
        else:
            self.logger.debug(f"[{name}] Sending termination signal.")
            if kill:
                os.kill(os.getpid(), signal.SIGTERM)

    def _register_signal_handlers(
        self,
        pool: Pool,
        manager: PipelineManager,
    ):
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
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            manager.terminate(force=True)
            self.logger.info("Server successfully exited.")
            self.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)
