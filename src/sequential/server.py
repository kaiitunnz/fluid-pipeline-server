import os
import signal
import time
from multiprocessing.pool import Pool, ThreadPool
from threading import Semaphore
from typing import Any, Optional

import numpy as np
from fluid_ai.pipeline import UiDetectionPipeline
from PIL import Image, ImageFile  # type: ignore

from src.benchmark import Benchmarker
from src.logger import DefaultLogger
from src.pipeline import PipelineModule
from src.sequential.manager import PipelineManager
from src.server import IPipelineServer, ServerCallbacks

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


class PipelineServer(IPipelineServer):
    """
    UI detection pipeline server with a sequential pipeline.

    It utilizes multithreading to handle client connections concurrently. It initializes
    exactly one instance of the UI detection pipeline, which executes on a single
    worker thread. Threads communicate via channels to send data to be processed
    and receive the results.

    Attributes
    ----------
    hostname : str
        Host name.
    port : int
        Port to listen to client connections.
    socket : Optional[sock.socket]
        Server socket.
    pipeline: UiDetectionPipeline
        Instance of the UI detection pipeline.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    num_workers : int
        Number of connection handling threads.
    verbose : bool
        Whether to log server events verbosely.
    test_mode : bool
        Whether to run the server in the test mode.
    logger : DefaultLogger
        Logger to log server events.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the server.
    manager : PipelineManager
        Manager of the UI detection pipeline.
    """

    pipeline: UiDetectionPipeline
    chunk_size: int
    max_image_size: int
    num_workers: int
    verbose: bool
    test_mode: bool
    logger: DefaultLogger
    benchmarker: Optional[Benchmarker]
    manager: PipelineManager
    _exit_sema: Semaphore  # To handle terminating signals only once

    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        pipeline: UiDetectionPipeline,
        callbacks: ServerCallbacks,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
        verbose: bool = True,
        benchmark: bool = False,
        benchmark_file: Optional[str] = None,
        test_mode: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        hostname : str
            Host name.
        port : int
            Port to listen to client connections.
        pipeline: UiDetectionPipeline
            Instance of the UI detection pipeline.
        callbacks: ServerCallbacks
            Server callbacks.
        chunk_size : int
            Chunk size for reading bytes from the sockets.
        max_image_size : int
            Maximum size of an image from the client.
        num_workers : int
            Number of connection handling threads.
        verbose : bool
            Whether to log server events verbosely.
        benchmark : bool
            Whether to run the server in the benchmark mode.
        benchmark_file : Optional[str]
            Path to the file to save the benchmark results.
        test_mode : bool
            Whether to run the server in the test mode.
        """
        super().__init__(
            hostname,
            port,
            None,
            DefaultLogger(PipelineServer._init_logger(verbose)),
            callbacks,
        )
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.test_mode = test_mode
        self._exit_sema = Semaphore(1)

        benchmark_metrics = [
            "Waiting time",
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

        self.manager = PipelineManager(
            pipeline, self.logger, self.benchmarker, self.getpid()
        )

        if len(kwargs) > 0:
            self.logger.warn(f"Got unexpected arguments: {kwargs}")

    def serve(self, warmup_image: Optional[str] = None):
        """Serves the UI detection pipeline server

        Parameters
        ----------
        warmup_image: Optional[str]
            Path to the image for warming up the UI detection pipeline and performing
            initial testing. `None` to skip the warming up stage.
        """
        self.logger.info("Starting the pipeline server...")

        self.manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool)

            if warmup_image is not None:
                try:
                    self.warmup(warmup_image)
                except Exception as e:
                    self.logger.error(f"[{self.manager.name}] Fatal error occured: {e}")
                    os.kill(self.getpid(), signal.SIGTERM)

            self.socket = self.bind()
            if self.socket is None:
                return

            job_no = 0
            while True:
                conn, addr = self.socket.accept()
                self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
                job_no += 1
                pool.apply_async(
                    self.manager.get_helper().serve,
                    args=(
                        job_no,
                        time.time(),
                        conn,
                        self.chunk_size,
                        self.max_image_size,
                        self.test_mode,
                    ),
                )

    def warmup(self, warmup_image: str, kill: bool = True):
        """Warms up the UI detection pipeline and performs initial testing

        Parameters
        ----------
        warmup_image : str
            Path to the image for warming up the UI detection pipeline and performing
            initial testing.
        kill : bool
            Whether to kill the process if an error occurs.
        """

        def on_error(message: str, info: Any):
            self.logger.error(f"{message} | Cause: {str(info)}")

        name = self.manager.name
        success = True

        self.logger.debug(f"[{name}] Warming up the pipeline...")
        img = np.asarray(Image.open(warmup_image))

        # Detect UI elements.
        detected = next(self.pipeline.detector([img]))
        self.logger.debug(f"[{name}] ({PipelineModule.DETECTOR.value}) PASSED.")

        # Filter UI elements.
        filtered = self.pipeline.filter(detected)
        self.logger.debug(f"[{name}] ({PipelineModule.FILTER.value}) PASSED.")

        # Match UI elements.
        matched = self.pipeline.matcher(filtered, filtered)
        if len(matched) == len(filtered):
            self.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) PASSED.")
        else:
            success = False
            self.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) FAILED.")
            on_error(
                "Failed to initialize the pipeline.",
                {
                    "detected": len(detected),
                    "filtered": len(filtered),
                    "matched": len(matched),
                },
            )

        # Partition the result.
        text_elems = []
        icon_elems = []
        for e in matched:
            if e.name in self.pipeline.textual_elements:
                text_elems.append(e)
            elif e.name in self.pipeline.icon_elements:
                icon_elems.append(e)

        # Extract UI info.
        self.pipeline.text_recognizer(text_elems)
        self.pipeline.icon_labeler(icon_elems)
        self.logger.debug(f"[{name}] ({PipelineModule.TEXT_RECOGNIZER.value}) PASSED.")
        self.logger.debug(f"[{name}] ({PipelineModule.ICON_LABELER.value}) PASSED.")

        # Extract UI relation.
        self.pipeline.relation(matched)
        self.logger.debug(f"[{name}] ({PipelineModule.RELATION.value}) PASSED.")

        if success:
            self.logger.debug(f"[{name}] Warm-up complete.")
        else:
            self.logger.debug(f"[{name}] Sending termination signal.")
            if kill:
                os.kill(os.getpid(), signal.SIGTERM)

    def _register_signal_handlers(self, pool: Pool):
        """Registers signal handlers to handle termination signals

        Parameters
        ----------
        pool : Pool
            Pool of connection handling threads.
        """
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.terminate()
            self.manager.terminate(force=True)
            if self.socket is not None:
                self.socket.close()
            self.logger.info("Server successfully exited.")
            self.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)
