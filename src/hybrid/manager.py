from typing import Optional


from src.constructor import PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.helper import PipelineHelper, PipelineManagerHelper
from src.hybrid.logging import Logger


class PipelineManager:
    key: int
    pipeline: PipelineConstructor
    helper: PipelineManagerHelper
    logger: Logger

    def __init__(
        self,
        key: int,
        pipeline: PipelineConstructor,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.pipeline = pipeline
        self.helper = PipelineManagerHelper(key, pipeline, logger, benchmarker)
        self.logger = logger

    def start(self):
        self.helper.start()

    def get_helper(self) -> PipelineHelper:
        return self.helper.get_helper()

    def terminate(self, force: bool = False):
        self.helper.terminate(force)
