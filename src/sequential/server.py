import logging
import os
import signal
import socket as sock
import sys
import time
from multiprocessing.pool import Pool, ThreadPool
from typing import Any, Optional

import numpy as np
from fluid_ai.pipeline import UiDetectionPipeline
from PIL import Image, ImageFile  # type: ignore

from src.benchmark import Benchmarker
from src.pipeline import IPipelineServer, PipelineModule
from src.process import UiDetectionArgs, ui_detection_serve
from src.sequential.helper import PipelineHelper
from src.sequential.manager import PipelineManager
from src.utils import ui_to_json

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


def _process(args: UiDetectionArgs) -> bytes:
    """Performs the UI detection process with a sequential pipeline

    Parameters
    ----------
    args : UiDetectionArgs
        Arguments. See `UiDetectionArgs` for more details.

    Returns
    -------
    bytes
        Result of the process, serialized into UTF-8-encoded JSON format.
    """
    helper, _, waiting_time, addr, screenshot_img, base_elements, _ = args
    assert isinstance(helper, PipelineHelper)

    # Process the screenshot.
    helper.log_debug(addr, "Processing UI elements.")
    processing_start = time.time()  # bench
    helper.send(PipelineModule.DETECTOR, screenshot_img, base_elements)
    results = helper.wait(PipelineModule.DETECTOR)
    processing_time = time.time() - processing_start  # bench
    helper.log_debug(addr, f"Found {len(results)} UI elements.")

    if helper.benchmarker is None:
        results_json = ui_to_json(screenshot_img, results).encode("utf-8")
    else:
        entry = [waiting_time, processing_time]  # type: ignore
        helper.benchmarker.add(entry)
        metrics = {"keys": helper.benchmarker.metrics, "values": entry}
        results_json = ui_to_json(screenshot_img, results, metrics=metrics).encode(
            "utf-8"
        )

    return results_json


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
    port : str
        Port to listen to client connections.
    pipeline: UiDetectionPipeline
        Instance of the UI detection pipeline.
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
    test_mode : bool
        Whether to run the server in the test mode.
    logger : logging.Logger
        Logger to log server events.
    benchmarker : Optional[Benchmarker]
        Benchmarker for benchmarking the server.
    manager : PipelineManager
        Manager of the UI detection pipeline.
    """

    hostname: str
    port: str
    pipeline: UiDetectionPipeline
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    verbose: bool
    test_mode: bool
    logger: logging.Logger
    benchmarker: Optional[Benchmarker]
    manager: PipelineManager

    def __init__(
        self,
        *,
        hostname: str,
        port: str,
        pipeline: UiDetectionPipeline,
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
        port : str
            Port to listen to client connections.
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
        benchmark : bool
            Whether to run the server in the benchmark mode.
        benchmark_file : Optional[str]
            Path to the file to save the benchmark results.
        test_mode : bool
            Whether to run the server in the test mode.
        """
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.test_mode = test_mode
        self.logger = PipelineServer._init_logger(verbose)

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

        self.manager = PipelineManager(pipeline, self.logger, self.benchmarker)

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

        if warmup_image is not None:
            self.warmup(warmup_image)

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        self.manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool)
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
                        self.manager.get_helper(),
                        job_no,
                        time.time(),
                        conn,
                        self.chunk_size,
                        self.max_image_size,
                        self.test_mode,
                        _process,
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
            self.logger.error(message)
            self.logger.debug("Cause:")
            self.logger.debug(str(info))

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
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            self.manager.terminate(force=True)
            self.logger.info("Server successfully exited.")
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)
