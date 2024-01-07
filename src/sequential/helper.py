import time
from queue import SimpleQueue
from typing import List, Optional, Tuple

from numpy import ndarray
from fluid_ai.base import UiElement

from src.benchmark import IBenchmarker
from src.helper import IPipelineHelper
from src.logger import ILogger
from src.pipeline import PipelineModule
from src.utils import ui_to_json


class PipelineHelper(IPipelineHelper):
    """
    Helper for accessing the UI detection pipeline.

    Attributes
    ----------
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    """

    _channel: SimpleQueue
    _res_channel: SimpleQueue

    def __init__(
        self,
        pipeline_ch: SimpleQueue,
        logger: ILogger,
        benchmarker: Optional[IBenchmarker],
        textual_elements: List[str],
        icon_elements: List[str],
    ):
        """
        Parameters
        ----------
        pipeline_ch : SimpleQueue
            Channel on which the pipeline worker listens for new jobs.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        """
        super().__init__(logger, benchmarker, textual_elements, icon_elements)
        self._channel = pipeline_ch
        self._res_channel = SimpleQueue()

    def send(self, _: PipelineModule, *args):
        self._channel.put((self._res_channel, args))

    def wait(self, _: PipelineModule) -> List[UiElement]:
        return self._res_channel.get()

    def process(
        self,
        _job_no: int,
        waiting_time: float,
        addr: Tuple[str, int],
        screenshot_img: ndarray,
        base_elements: Optional[List[UiElement]],
        _test_mode: bool,
    ):
        """Executes the UI detection process with a sequential pipeline

        Parameters
        ----------
        job_no : int
            Job number, used to identify the job.
        waiting_time : float
            Time for which the client waits until its request gets handled.
        addr : Tuple[str, int]
            Client's IP address.
        screenshot_img : ndarray
            Screenshot to be processed.
        base_elements : Optional[List[UiElement]]
            Base elements, aka additional UI elements.
        test_mode : bool
            Whether to handle connections in test mode.

        Returns
        -------
        bytes
            Result of the process, serialized into UTF-8-encoded JSON format.
        """
        # Process the screenshot.
        self.log_debug(addr, "Processing UI elements.")
        processing_start = time.time()  # bench
        self.send(PipelineModule.DETECTOR, screenshot_img, base_elements)
        results = self.wait(PipelineModule.DETECTOR)
        processing_time = time.time() - processing_start  # bench
        self.log_debug(addr, f"Found {len(results)} UI elements.")

        if self.benchmarker is None:
            results_json = ui_to_json(screenshot_img, results).encode("utf-8")
        else:
            entry = [waiting_time, processing_time]  # type: ignore
            self.benchmarker.add(entry)
            metrics = {"keys": self.benchmarker.metrics, "values": entry}
            results_json = ui_to_json(screenshot_img, results, metrics=metrics).encode(
                "utf-8"
            )

        return results_json
