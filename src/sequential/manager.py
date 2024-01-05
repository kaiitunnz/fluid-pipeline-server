from queue import SimpleQueue
from typing import Any, List, Optional

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.pipeline import UiDetectionPipeline

from src.benchmark import Benchmarker
from src.logger import DefaultLogger
from src.sequential.helper import PipelineHelper
from src.sequential.worker import Worker


def detect(
    screenshot: np.ndarray, base_elements: List[UiElement], module: Any
) -> List[UiElement]:
    """Executes the UI detection pipeline

    Parameters
    ----------
    screenshot : ndarray
        Screenshot to be processed.
    base_elements : List[UiElement]
        Base UI elements, aka additional UI elements.
    module : Any
        UI detection pipeline.

    Returns
    -------
    List[UiElement]
        Resulting list of UI elements.
    """
    assert isinstance(module, UiDetectionPipeline)
    return next(module.detect([screenshot], [base_elements]))


class PipelineManager:
    """
    Manager of the UI detection pipeline.

    It manages all the resources used by the UI detection pipeline, especially the
    pipeline worker thread.

    Attributes
    ----------
    pipeline : UiDetectionPipeline
        Instance of the UI detection pipeline.
    pipeline_ch : SimpleQueue
        Channel on which the pipeline worker listens for new jobs.
    worker : Worker
        Pipeline worker.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server.
    name : str
        Name of the instance, used to identify itself in the server log.
    """

    pipeline: UiDetectionPipeline
    pipeline_ch: SimpleQueue
    worker: Worker
    logger: DefaultLogger
    benchmarker: Optional[Benchmarker]
    name: str

    def __init__(
        self,
        pipeline: UiDetectionPipeline,
        logger: DefaultLogger,
        benchmarker: Optional[Benchmarker],
        name: str = "Pipeline",
    ):
        """
        Parameters
        ----------
        pipeline : UiDetectionPipeline
            Instance of the UI detection pipeline.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server.
        name : str
            Name of the instance, used to identify itself in the server log.
        """
        self.pipeline = pipeline
        self.pipeline_ch = SimpleQueue()
        self.logger = logger
        self.benchmarker = benchmarker
        self.name = name

    def start(self):
        """Starts the UI detection pipeline's worker thread"""
        self.worker = Worker(
            detect, self.pipeline_ch, self.pipeline, self.logger, self.name
        )
        self.worker.start()

    def get_helper(self):
        """Instantiates a pipeline helper which can be used by connection handling
        threads to access the UI detection pipeline

        Returns
        -------
        PipelineHelper
            Helper for accessing the UI detection pipeline.
        """
        return PipelineHelper(
            self.pipeline_ch,
            self.logger,
            self.benchmarker,
            self.pipeline.textual_elements,
            self.pipeline.icon_elements,
        )

    def terminate(self, force: bool = False):
        """Terminates the UI detection pipeline's worker thread

        Parameters
        ----------
        force : bool
            Whether to immediately terminate the worker thread without waiting for
            the pending jobs to finish.
        """
        self.logger.info("Terminating the worker processes...")
        self.worker.terminate(force)
        self.logger.info("Done.")
