import logging
import os
import signal
import socket as sock
import sys
import time
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as tmp
from PIL import Image, ImageFile
from fluid_ai.base import UiElement
from multiprocessing.managers import SyncManager
from multiprocessing.pool import Pool

from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.multiprocessing.manager import PipelineHelper, PipelineManager
from src.utils import readall, ui_to_json

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"
SAVE_IMG_DIR = "res"
SAVE_IMG = False


def _handle_connection(
    conn: sock.socket,
    helper: PipelineHelper,
    chunk_size: int,
    max_image_size: int,
    start_time: float,
):
    waiting_time = time.time() - start_time  # bench

    try:
        addr = conn.getpeername()
    except Exception as e:
        helper.logger.error(
            f"The following exception occurred while attempting to connect: {e}"
        )
        return

    try:
        packet_size = int.from_bytes(readall(conn, 4, chunk_size), "big", signed=False)
        helper.log_debug(addr, f"Receiving a packet of size {packet_size} bytes.")
        if packet_size > max_image_size:
            helper.log_error(
                addr, f"The packet size exceeds the maximum allowable size."
            )
            conn.close()
            return

        data = readall(conn, packet_size, chunk_size)
        helper.log_debug(addr, f"Received a packet of size {len(data)} bytes.")
        packet = BytesIO(data)
        img = Image.open(packet)
        if SAVE_IMG:
            helper.save_image(img, SAVE_IMG_DIR)
        screenshot_img = torch.tensor(np.asarray(img))
        helper.log_debug(
            addr, f"Received an image of shape {tuple(screenshot_img.size())}."
        )

        # Detect UI elements.
        helper.log_debug(addr, "Detecting UI elements.")
        detection_start = time.time()  # bench
        helper.detect(screenshot_img)
        detected = helper.wait_detect()
        detection_time = time.time() - detection_start  # bench
        helper.log_debug(addr, f"Found {len(detected)} UI elements.")

        # Partition the result.
        text_elems = []
        icon_elems = []
        results: List[UiElement] = []
        for e in detected:
            if e.name in helper.textual_elements:
                text_elems.append(e)
            elif e.name in helper.icon_elements:
                icon_elems.append(e)
            else:
                results.append(e)

        # Extract UI info.
        helper.log_debug(addr, "Extracting UI info.")
        if helper.benchmarker is None:
            helper.recognize_text(text_elems)
            helper.label_icons(icon_elems)
            results.extend(helper.wait_recognize_text())
            results.extend(helper.wait_label_icons())
        else:
            text_start = time.time()  # bench
            helper.recognize_text(text_elems)
            results.extend(helper.wait_recognize_text())
            text_time = time.time() - text_start  # bench
            icon_start = time.time()  # bench
            helper.label_icons(icon_elems)
            results.extend(helper.wait_label_icons())
            icon_time = time.time() - icon_start  # bench

        processing_time = time.time() - detection_start  # bench
        if helper.benchmarker is None:
            results_json = ui_to_json(screenshot_img.numpy(), results).encode("utf-8")
        else:
            entry = [waiting_time, detection_time, text_time, icon_time, processing_time]  # type: ignore
            helper.benchmarker.add(entry)
            metrics = {"keys": helper.benchmarker.metrics, "values": entry}
            results_json = ui_to_json(
                screenshot_img.numpy(), results, metrics=metrics
            ).encode("utf-8")

        helper.log_debug(
            addr,
            f"Sending back the response of size {len(results_json)} bytes.",
        )

        conn.sendall(len(results_json).to_bytes(4, "big", signed=False))
        helper.log_debug(addr, f"Sent response size: {len(results_json)}")
        conn.sendall(results_json)
        helper.log_info(addr, "Response sent.")
    except Exception as e:
        helper.log_error(addr, f"The following exception occurred: {e}")
    finally:
        helper.log_info(addr, "Connection closed.")
        conn.close()


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

    def __init__(
        self,
        *,
        hostname: str,
        port: str,
        pipeline: PipelineConstructor,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
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
        self.verbose = verbose
        self.logger = PipelineServer._init_logger(verbose)

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

        # Required to prevent PyTorch from hanging
        # https://github.com/pytorch/pytorch/issues/82843
        torch.set_num_threads(1)

        if torch.cuda.is_available():
            tmp.set_start_method("forkserver")  # To enable CUDA.

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)

        with tmp.Manager() as sync_manager:
            manager = PipelineManager(
                self.pipeline, sync_manager, self.logger, self.benchmarker
            )
            manager.start()

            if warmup_image is not None:
                self._warmup(manager.get_helper(), warmup_image)

            with tmp.Pool(processes=self.num_workers) as pool:
                self._register_signal_handlers(pool, sync_manager, manager)
                self.logger.info(
                    f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
                )
                while True:
                    conn, addr = self.socket.accept()
                    self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
                    pool.apply_async(
                        _handle_connection,
                        args=(
                            conn,
                            manager.get_helper(),
                            self.chunk_size,
                            self.max_image_size,
                            time.time(),
                        ),
                    )

    def _warmup(self, helper: PipelineHelper, warmup_image: str):
        self.logger.info("Warming up the pipeline...")
        img = Image.open(warmup_image)
        screenshot_img = torch.tensor(np.asarray(img))

        # Detect UI elements.
        helper.detect(screenshot_img)
        detected = helper.wait_detect()

        # Partition the result.
        text_elems = []
        icon_elems = []
        for e in detected:
            if e.name in helper.textual_elements:
                text_elems.append(e)
            elif e.name in helper.icon_elements:
                icon_elems.append(e)

        # Extract UI info.
        helper.recognize_text(text_elems)
        helper.label_icons(icon_elems)
        helper.wait_recognize_text()
        helper.wait_label_icons()

    def _register_signal_handlers(
        self,
        pool: Pool,
        sync_manager: SyncManager,
        manager: PipelineManager,
    ):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            manager.terminate(force=True)
            sync_manager.shutdown()
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
