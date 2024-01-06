from typing import Optional

from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.logger import DefaultLogger
from src.multithread.helper import PipelineHelper, PipelineManagerHelper, WarmupHelper


class PipelineManager:
    """
    Manager of the UI detection pipeline.

    It manages all the resources used by the UI detection pipeline, especially the
    worker threads.

    Attributes
    ----------
    pipeline : PipelineConstructor
        Constructor of the UI detection pipeline.
    helper: PipelineManagerHelper
        Helper that manages the worker threads to serve the UI detection pipeline.
    logger : Logger
        Logger to log the UI detection process.
    """

    pipeline: PipelineConstructor
    helper: PipelineManagerHelper
    logger: DefaultLogger

    def __init__(
        self,
        pipeline: PipelineConstructor,
        logger: DefaultLogger,
        benchmarker: Optional[Benchmarker],
        server_pid: int,
        num_instances: int,
    ):
        """
        Parameters
        ----------
        pipeline : PipelineConstructor
            Constructor of the UI detection pipeline.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        server_pid : int
            Process ID of the pipeline server.
        num_instances : int
            Number of instances of the UI detection pipeline to be created.
        """
        self.pipeline = pipeline
        self.helper = PipelineManagerHelper(
            pipeline, logger, benchmarker, num_instances, server_pid
        )
        self.logger = logger

    def start(self):
        """Starts the UI detection pipeline's worker threads"""
        self.helper.start()

    def get_helper(self) -> PipelineHelper:
        """Instantiates a pipeline helper which can be used by connection handling
        threads to access the UI detection pipeline modules

        Returns
        -------
        PipelineHelper
            Helper for accessing the UI detection pipeline modules.
        """
        return self.helper.get_helper()

    def get_warmup_helper(self, i: int) -> WarmupHelper:
        """Instantiates a pipeline helper which can be used to warm up the UI detection
        pipeline.

        Returns
        -------
        i : int
            Index of the instance of the UI detection pipeline to be warmed up.
        """
        return self.helper.get_warmup_helper(i)

    def terminate(self, force: bool = False):
        """Terminates the UI detection pipeline's worker threads"""
        self.logger.info("Terminating the workers...")
        self.helper.terminate(force)
