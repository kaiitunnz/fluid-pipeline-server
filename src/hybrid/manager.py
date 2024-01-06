from typing import Optional

from src.constructor import PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.helper import PipelineHelper, PipelineManagerHelper
from src.hybrid.logger import Logger


class PipelineManager:
    """
    Manager of the UI detection pipeline.

    It manages all the resources used by the UI detection pipeline, especially the
    worker threads.

    Attributes
    ----------
    key : int
        ID of the instance. It is used to differentiate the instance from other instances
        of this class.
    pipeline : PipelineConstructor
        Constructor of the UI detection pipeline.
    helper: PipelineManagerHelper
        Helper that manages the worker threads to serve the UI detection pipeline.
    logger : Logger
        Logger to log the UI detection process.
    """

    key: int
    pipeline: PipelineConstructor
    helper: PipelineManagerHelper
    logger: Logger

    _server_pid: int

    def __init__(
        self,
        key: int,
        pipeline: PipelineConstructor,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        server_pid: int,
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other
            instances of this class.
        pipeline : PipelineConstructor
            Constructor of the UI detection pipeline.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        server_pid : int
            Process ID of the pipeline server.
        """
        self.pipeline = pipeline
        self.helper = PipelineManagerHelper(
            key, pipeline, logger, benchmarker, server_pid
        )
        self.logger = logger
        self._server_pid = server_pid

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

    def terminate(self, force: bool = False):
        """Terminates the UI detection pipeline's worker threads

        It waits until all the workers finish their current jobs.
        """
        self.helper.terminate(force)
