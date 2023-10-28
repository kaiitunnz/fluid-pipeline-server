import logging
from typing import Optional


from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.threading.helper import PipelineHelper, PipelineManagerHelper


class PipelineManager:
    pipeline: PipelineConstructor
    helper: PipelineManagerHelper
    logger: logging.Logger

    def __init__(
        self,
        pipeline: PipelineConstructor,
        logger: logging.Logger,
        benchmarker: Optional[Benchmarker],
        num_instances: int,
    ):
        self.pipeline = pipeline
        self.helper = PipelineManagerHelper(
            pipeline, logger, benchmarker, num_instances
        )
        self.logger = logger

    def start(self):
        self.helper.start()

    def get_helper(self) -> PipelineHelper:
        return self.helper.get_helper()

    def terminate(self, force: bool = False):
        self.logger.info("Terminating the workers...")
        self.helper.terminate()
